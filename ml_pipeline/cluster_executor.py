"""
Cluster Executor for GeneX Project

Manages execution of ML/DL/AI pipelines on HPC clusters (Graham, Beluga, Niagara)
for scalable processing of scientific papers and knowledge extraction.
"""

import logging
import subprocess
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import paramiko
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ClusterConfig:
    """Configuration for cluster execution"""
    cluster_name: str  # graham, beluga, niagara
    username: str
    project_dir: str
    scratch_dir: str
    max_jobs: int = 10
    job_timeout: int = 3600  # 1 hour
    memory_per_job: str = "4G"
    cpus_per_job: int = 4
    gpu_per_job: int = 1

@dataclass
class JobSpec:
    """Specification for a cluster job"""
    job_id: str
    script_path: str
    input_data: Dict[str, Any]
    output_dir: str
    dependencies: List[str] = None
    priority: int = 1

class ClusterExecutor:
    """
    Manages execution of GeneX ML/DL/AI pipelines on HPC clusters.
    Handles job submission, monitoring, and result collection.
    """

    def __init__(self, config: ClusterConfig):
        self.config = config
        self.ssh_config = self._load_ssh_config()
        self.active_jobs = {}
        self.job_queue = []

        logger.info(f"Cluster executor initialized for {config.cluster_name}")

    def _load_ssh_config(self) -> Dict[str, Any]:
        """Load SSH configuration for cluster connections"""
        ssh_config_path = Path.home() / ".ssh" / "config"

        if not ssh_config_path.exists():
            raise FileNotFoundError("SSH config not found. Run setup_hpc_connections.sh first.")

        # Parse SSH config (simplified)
        config = {}
        current_host = None

        with open(ssh_config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Host '):
                    current_host = line.split()[1]
                    config[current_host] = {}
                elif current_host and ' ' in line:
                    key, value = line.split(maxsplit=1)
                    config[current_host][key] = value

        return config

    def submit_ml_pipeline_job(self,
                              input_papers: List[Dict[str, Any]],
                              pipeline_type: str = "comprehensive",
                              output_dir: str = None) -> str:
        """
        Submit ML/DL/AI pipeline job to cluster.

        Args:
            input_papers: List of paper data to process
            pipeline_type: Type of pipeline (comprehensive, classification, quality, knowledge)
            output_dir: Output directory for results

        Returns:
            Job ID
        """
        job_id = f"genex_ml_{pipeline_type}_{int(time.time())}"

        # Create job script
        script_content = self._create_ml_job_script(
            job_id, input_papers, pipeline_type, output_dir
        )

        # Save script to local temp directory
        script_path = f"temp_jobs/{job_id}.sh"
        os.makedirs("temp_jobs", exist_ok=True)

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Create job specification
        job_spec = JobSpec(
            job_id=job_id,
            script_path=script_path,
            input_data={'papers': input_papers, 'pipeline_type': pipeline_type},
            output_dir=output_dir or f"results/{job_id}"
        )

        # Submit job to cluster
        self._submit_job_to_cluster(job_spec)

        return job_id

    def _create_ml_job_script(self, job_id: str, input_papers: List[Dict[str, Any]],
                             pipeline_type: str, output_dir: str) -> str:
        """Create SLURM job script for ML pipeline execution"""

        # Determine resource requirements based on pipeline type
        if pipeline_type == "comprehensive":
            time_limit = "4:00:00"
            memory = "32G"
            cpus = 8
            gpus = 2
        elif pipeline_type == "classification":
            time_limit = "2:00:00"
            memory = "16G"
            cpus = 4
            gpus = 1
        elif pipeline_type == "quality":
            time_limit = "2:00:00"
            memory = "16G"
            cpus = 4
            gpus = 1
        elif pipeline_type == "knowledge":
            time_limit = "3:00:00"
            memory = "24G"
            cpus = 6
            gpus = 1
        else:
            time_limit = "2:00:00"
            memory = "16G"
            cpus = 4
            gpus = 1

        script = f"""#!/bin/bash
#SBATCH --job-name={job_id}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --account=def-{self.config.username}
#SBATCH --output=logs/{job_id}_%j.out
#SBATCH --error=logs/{job_id}_%j.err

# Load required modules
module load python/3.9
module load cuda/11.4
module load cudnn/8.2

# Activate virtual environment
source ~/genex_env/bin/activate

# Set up working directory
cd {self.config.project_dir}

# Create output directory
mkdir -p {output_dir}
mkdir -p logs

# Copy input data to scratch
SCRATCH_DIR="$SCRATCH/genex/{job_id}"
mkdir -p $SCRATCH_DIR

# Save input papers to JSON
cat > $SCRATCH_DIR/input_papers.json << 'EOF'
{json.dumps(input_papers, indent=2)}
EOF

# Run ML pipeline
echo "Starting {pipeline_type} ML pipeline for {len(input_papers)} papers"
python -m src.ml_pipeline.cluster_runner \\
    --job-id {job_id} \\
    --pipeline-type {pipeline_type} \\
    --input-file $SCRATCH_DIR/input_papers.json \\
    --output-dir {output_dir} \\
    --scratch-dir $SCRATCH_DIR

# Copy results back
cp -r $SCRATCH_DIR/results/* {output_dir}/

# Clean up scratch
rm -rf $SCRATCH_DIR

echo "Job {job_id} completed successfully"
"""

        return script

    def _submit_job_to_cluster(self, job_spec: JobSpec):
        """Submit job to cluster using SLURM"""

        # Copy script to cluster
        remote_script_path = f"{self.config.project_dir}/jobs/{job_spec.job_id}.sh"

        # Use scp to copy script
        scp_cmd = [
            "scp", job_spec.script_path,
            f"{self.config.cluster_name}:{remote_script_path}"
        ]

        try:
            subprocess.run(scp_cmd, check=True, capture_output=True)
            logger.info(f"Script copied to cluster: {remote_script_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy script to cluster: {e}")
            raise

        # Submit job using ssh
        ssh_cmd = [
            "ssh", self.config.cluster_name,
            f"cd {self.config.project_dir} && sbatch {remote_script_path}"
        ]

        try:
            result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split()[-1]  # Extract SLURM job ID
            job_spec.slurm_job_id = job_id
            self.active_jobs[job_spec.job_id] = job_spec
            logger.info(f"Job submitted successfully: {job_spec.job_id} (SLURM: {job_id})")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e}")
            raise

    def monitor_jobs(self) -> Dict[str, str]:
        """
        Monitor active jobs and return their status.

        Returns:
            Dictionary mapping job IDs to status
        """
        if not self.active_jobs:
            return {}

        # Get job status from cluster
        job_ids = [job.slurm_job_id for job in self.active_jobs.values() if hasattr(job, 'slurm_job_id')]

        if not job_ids:
            return {}

        squeue_cmd = ["ssh", self.config.cluster_name, "squeue", "--jobs", ",".join(job_ids)]

        try:
            result = subprocess.run(squeue_cmd, check=True, capture_output=True, text=True)
            status_lines = result.stdout.strip().split('\n')[1:]  # Skip header

            status_map = {}
            for line in status_lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        slurm_job_id = parts[0]
                        status = parts[4]
                        status_map[slurm_job_id] = status

            # Map back to our job IDs
            job_status = {}
            for job_id, job_spec in self.active_jobs.items():
                if hasattr(job_spec, 'slurm_job_id'):
                    slurm_status = status_map.get(job_spec.slurm_job_id, 'UNKNOWN')
                    job_status[job_id] = slurm_status
                else:
                    job_status[job_id] = 'SUBMITTING'

            return job_status

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get job status: {e}")
            return {job_id: 'UNKNOWN' for job_id in self.active_jobs.keys()}

    def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve results for a completed job.

        Args:
            job_id: Job ID

        Returns:
            Job results or None if not completed
        """
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return None

        job_spec = self.active_jobs[job_id]

        # Check if job is completed
        status = self.monitor_jobs().get(job_id, 'UNKNOWN')
        if status not in ['COMPLETED', 'CANCELLED', 'FAILED']:
            logger.info(f"Job {job_id} still running (status: {status})")
            return None

        # Download results
        local_results_dir = f"results/{job_id}"
        remote_results_dir = f"{self.config.project_dir}/{job_spec.output_dir}"

        # Use rsync to download results
        rsync_cmd = [
            "rsync", "-avz", "--progress",
            f"{self.config.cluster_name}:{remote_results_dir}/",
            f"{local_results_dir}/"
        ]

        try:
            subprocess.run(rsync_cmd, check=True, capture_output=True)
            logger.info(f"Results downloaded for job {job_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download results for job {job_id}: {e}")
            return None

        # Load results
        results_file = f"{local_results_dir}/results.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)

            # Remove from active jobs
            del self.active_jobs[job_id]

            return results
        else:
            logger.warning(f"Results file not found for job {job_id}")
            return None

    def submit_batch_processing(self,
                               papers_batch: List[List[Dict[str, Any]]],
                               pipeline_type: str = "comprehensive") -> List[str]:
        """
        Submit multiple jobs for batch processing.

        Args:
            papers_batch: List of paper batches
            pipeline_type: Type of pipeline to run

        Returns:
            List of job IDs
        """
        job_ids = []

        for i, papers in enumerate(papers_batch):
            job_id = self.submit_ml_pipeline_job(
                input_papers=papers,
                pipeline_type=pipeline_type,
                output_dir=f"results/batch_{i}_{pipeline_type}"
            )
            job_ids.append(job_id)

            # Add delay to avoid overwhelming the scheduler
            time.sleep(2)

        logger.info(f"Submitted {len(job_ids)} batch jobs")
        return job_ids

    def wait_for_completion(self, job_ids: List[str], timeout: int = None) -> Dict[str, Any]:
        """
        Wait for jobs to complete and collect all results.

        Args:
            job_ids: List of job IDs to wait for
            timeout: Maximum time to wait (seconds)

        Returns:
            Dictionary mapping job IDs to results
        """
        start_time = time.time()
        results = {}

        while job_ids:
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout reached while waiting for jobs")
                break

            # Check job status
            status = self.monitor_jobs()

            for job_id in job_ids[:]:  # Copy list to avoid modification during iteration
                job_status = status.get(job_id, 'UNKNOWN')

                if job_status == 'COMPLETED':
                    # Get results
                    job_results = self.get_job_results(job_id)
                    if job_results:
                        results[job_id] = job_results
                        job_ids.remove(job_id)
                        logger.info(f"Job {job_id} completed and results collected")

                elif job_status in ['CANCELLED', 'FAILED']:
                    logger.warning(f"Job {job_id} failed with status: {job_status}")
                    results[job_id] = {'error': f'Job failed with status: {job_status}'}
                    job_ids.remove(job_id)

            if job_ids:
                logger.info(f"Waiting for {len(job_ids)} jobs to complete...")
                time.sleep(30)  # Check every 30 seconds

        return results

    def cancel_job(self, job_id: str):
        """Cancel a running job."""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return

        job_spec = self.active_jobs[job_id]

        if hasattr(job_spec, 'slurm_job_id'):
            scancel_cmd = ["ssh", self.config.cluster_name, "scancel", job_spec.slurm_job_id]

            try:
                subprocess.run(scancel_cmd, check=True, capture_output=True)
                logger.info(f"Job {job_id} cancelled successfully")
                del self.active_jobs[job_id]
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to cancel job {job_id}: {e}")

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about cluster status and resources."""
        try:
            # Get queue information
            squeue_cmd = ["ssh", self.config.cluster_name, "squeue", "--format", "%i %j %u %t %M %N"]
            result = subprocess.run(squeue_cmd, check=True, capture_output=True, text=True)

            # Parse queue information
            queue_info = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:
                        queue_info.append({
                            'job_id': parts[0],
                            'name': parts[1],
                            'user': parts[2],
                            'status': parts[3],
                            'time': parts[4],
                            'nodes': parts[5]
                        })

            # Get cluster load
            sinfo_cmd = ["ssh", self.config.cluster_name, "sinfo", "--format", "%N %T %C %m"]
            result = subprocess.run(sinfo_cmd, check=True, capture_output=True, text=True)

            # Parse cluster information
            cluster_info = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        cluster_info.append({
                            'nodes': parts[0],
                            'state': parts[1],
                            'cpus': parts[2],
                            'memory': parts[3]
                        })

            return {
                'queue_info': queue_info,
                'cluster_info': cluster_info,
                'active_jobs': len(self.active_jobs)
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {'error': str(e)}
