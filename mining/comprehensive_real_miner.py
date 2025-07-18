import requests
import aiohttp
import asyncio
import os
import json
import logging
from typing import List, Dict
from datetime import datetime
import yaml

# Import base miner class
try:
    from .base_miner import BaseMiner
except ImportError:
    # Fallback if base_miner doesn't exist
    class BaseMiner:
        """Base class for all miners."""
        def __init__(self, config: dict):
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__)

class UCSCMiner(BaseMiner):
    """Real UCSC Genome Browser data miner for genomic annotations and tracks."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = "https://api.genome.ucsc.edu"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GeneX-Mega-Project/1.0 (Academic Research)'
        })

    async def mine_genome_annotations(self, genome: str = "hg38") -> List[Dict]:
        """Mine genome annotations from UCSC Genome Browser."""
        try:
            # Get available tracks
            tracks_url = f"{self.base_url}/list/tracks"
            params = {"genome": genome}

            async with aiohttp.ClientSession() as session:
                async with session.get(tracks_url, params=params) as response:
                    if response.status == 200:
                        tracks_data = await response.json()

                        annotations = []
                        for track in tracks_data.get("tracks", [])[:50]:  # Limit for demo
                            track_name = track.get("track")
                            if track_name:
                                # Get track details
                                track_url = f"{self.base_url}/getData/track"
                                track_params = {
                                    "genome": genome,
                                    "track": track_name,
                                    "maxItemsOutput": 100
                                }

                                async with session.get(track_url, params=track_params) as track_response:
                                    if track_response.status == 200:
                                        track_data = await track_response.json()
                                        annotations.append({
                                            "source": "UCSC",
                                            "genome": genome,
                                            "track": track_name,
                                            "data": track_data,
                                            "mined_at": datetime.now().isoformat()
                                        })

                        return annotations
                    else:
                        self.logger.error(f"UCSC API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error mining UCSC data: {e}")
            return []


class EnsemblMiner(BaseMiner):
    """Real Ensembl data miner for gene annotations and variants."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = "https://rest.ensembl.org"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'GeneX-Mega-Project/1.0 (Academic Research)'
        })

    async def mine_gene_annotations(self, species: str = "homo_sapiens") -> List[Dict]:
        """Mine gene annotations from Ensembl."""
        try:
            # Get gene list
            genes_url = f"{self.base_url}/overlap/region/{species}/13:32315474-32400266"
            params = {"feature": "gene"}

            async with aiohttp.ClientSession() as session:
                async with session.get(genes_url, params=params) as response:
                    if response.status == 200:
                        genes_data = await response.json()

                        annotations = []
                        for gene in genes_data[:20]:  # Limit for demo
                            gene_id = gene.get("id")
                            if gene_id:
                                # Get detailed gene info
                                gene_url = f"{self.base_url}/lookup/{species}/{gene_id}"

                                async with session.get(gene_url) as gene_response:
                                    if gene_response.status == 200:
                                        gene_info = await gene_response.json()
                                        annotations.append({
                                            "source": "Ensembl",
                                            "species": species,
                                            "gene_id": gene_id,
                                            "data": gene_info,
                                            "mined_at": datetime.now().isoformat()
                                        })

                        return annotations
                    else:
                        self.logger.error(f"Ensembl API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error mining Ensembl data: {e}")
            return []


class GEODataMiner(BaseMiner):
    """Real GEO (Gene Expression Omnibus) data miner."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.api_key = config.get("ncbi_api_key", "")

    async def mine_expression_data(self, query: str = "CRISPR") -> List[Dict]:
        """Mine gene expression data from GEO."""
        try:
            # Search for datasets
            search_url = f"{self.base_url}/esearch.fcgi"
            params = {
                "db": "gds",
                "term": query,
                "retmax": 50,
                "retmode": "json"
            }
            if self.api_key:
                params["api_key"] = self.api_key

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        search_data = await response.json()
                        id_list = search_data.get("esearchresult", {}).get("idlist", [])

                        datasets = []
                        for gds_id in id_list[:10]:  # Limit for demo
                            # Get dataset summary
                            summary_url = f"{self.base_url}/esummary.fcgi"
                            summary_params = {
                                "db": "gds",
                                "id": gds_id,
                                "retmode": "json"
                            }
                            if self.api_key:
                                summary_params["api_key"] = self.api_key

                            async with session.get(summary_url, params=summary_params) as summary_response:
                                if summary_response.status == 200:
                                    summary_data = await summary_response.json()
                                    datasets.append({
                                        "source": "GEO",
                                        "gds_id": gds_id,
                                        "data": summary_data,
                                        "mined_at": datetime.now().isoformat()
                                    })

                        return datasets
                    else:
                        self.logger.error(f"GEO API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error mining GEO data: {e}")
            return []


class UniProtMiner(BaseMiner):
    """Real UniProt data miner for protein annotations."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = "https://rest.uniprot.org"

    async def mine_protein_annotations(self, query: str = "CRISPR") -> List[Dict]:
        """Mine protein annotations from UniProt."""
        try:
            # Search for proteins
            search_url = f"{self.base_url}/uniprotkb/search"
            params = {
                "query": query,
                "format": "json",
                "size": 50
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        search_data = await response.json()
                        results = search_data.get("results", [])

                        annotations = []
                        for protein in results[:20]:  # Limit for demo
                            accession = protein.get("primaryAccession")
                            if accession:
                                # Get detailed protein info
                                protein_url = f"{self.base_url}/uniprotkb/{accession}"

                                async with session.get(protein_url) as protein_response:
                                    if protein_response.status == 200:
                                        protein_data = await protein_response.json()
                                        annotations.append({
                                            "source": "UniProt",
                                            "accession": accession,
                                            "data": protein_data,
                                            "mined_at": datetime.now().isoformat()
                                        })

                        return annotations
                    else:
                        self.logger.error(f"UniProt API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error mining UniProt data: {e}")
            return []


class PDBMiner(BaseMiner):
    """Real PDB (Protein Data Bank) data miner for structural data."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = "https://data.rcsb.org/rest/v1/core"

    async def mine_structure_data(self, query: str = "CRISPR") -> List[Dict]:
        """Mine protein structure data from PDB."""
        try:
            # Search for structures
            search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
            search_data = {
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {
                        "value": query
                    }
                },
                "return_type": "entry",
                "request_options": {
                    "pager": {
                        "start": 0,
                        "rows": 50
                    }
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(search_url, json=search_data) as response:
                    if response.status == 200:
                        search_results = await response.json()
                        entries = search_results.get("total_count", 0)

                        structures = []
                        if entries > 0:
                            # Get structure details for first few results
                            pdb_ids = search_results.get("result_set", [])[:10]

                            for pdb_id in pdb_ids:
                                structure_url = f"{self.base_url}/entry/{pdb_id}"

                                async with session.get(structure_url) as structure_response:
                                    if structure_response.status == 200:
                                        structure_data = await structure_response.json()
                                        structures.append({
                                            "source": "PDB",
                                            "pdb_id": pdb_id,
                                            "data": structure_data,
                                            "mined_at": datetime.now().isoformat()
                                        })

                        return structures
                    else:
                        self.logger.error(f"PDB API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error mining PDB data: {e}")
            return []


class ClinicalTrialsMiner(BaseMiner):
    """Real ClinicalTrials.gov data miner for clinical trial data."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = "https://classic.clinicaltrials.gov/api/query/study_fields"

    async def mine_clinical_trials(self, query: str = "CRISPR") -> List[Dict]:
        """Mine clinical trial data from ClinicalTrials.gov."""
        try:
            # Search for clinical trials
            params = {
                "expr": query,
                "fields": "NCTId,BriefTitle,OfficialTitle,Condition,InterventionName,Phase,Status,LeadSponsorName,StartDate,CompletionDate,EnrollmentCount,StudyType",
                "min_rnk": 1,
                "max_rnk": 50,
                "fmt": "json"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        trials_data = await response.json()
                        studies = trials_data.get("StudyFieldsResponse", {}).get("StudyFields", [])

                        trials = []
                        for study in studies:
                            trial_info = {
                                "source": "ClinicalTrials.gov",
                                "nct_id": study.get("NCTId", [""])[0] if study.get("NCTId") else "",
                                "title": study.get("BriefTitle", [""])[0] if study.get("BriefTitle") else "",
                                "condition": study.get("Condition", []),
                                "intervention": study.get("InterventionName", []),
                                "phase": study.get("Phase", []),
                                "status": study.get("Status", [""])[0] if study.get("Status") else "",
                                "sponsor": study.get("LeadSponsorName", [""])[0] if study.get("LeadSponsorName") else "",
                                "enrollment": study.get("EnrollmentCount", [""])[0] if study.get("EnrollmentCount") else "",
                                "mined_at": datetime.now().isoformat()
                            }
                            trials.append(trial_info)

                        return trials
                    else:
                        self.logger.error(f"ClinicalTrials.gov API error: {response.status}")
                        return []

        except Exception as e:
            self.logger.error(f"Error mining ClinicalTrials.gov data: {e}")
            return []


class ComprehensiveRealMiner:
    """Comprehensive real data miner orchestrating all mining operations."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Initialize all available miners (removed non-existent ones)
        self.encode_miner = ENCODEMiner(self.config)
        self.ucsc_miner = UCSCMiner(self.config)
        self.ensembl_miner = EnsemblMiner(self.config)
        self.geo_miner = GEODataMiner(self.config)
        self.uniprot_miner = UniProtMiner(self.config)
        self.pdb_miner = PDBMiner(self.config)
        self.clinical_miner = ClinicalTrialsMiner(self.config)

        # Project-specific queries
        self.project_queries = {
            "crispr_tool_development": ["CRISPR tool development", "CRISPR-Cas9", "CRISPR-Cas12"],
            "prime_editing": ["Prime editing", "PE3", "PE5", "prime editor"],
            "base_editing": ["Base editing", "CBE", "ABE", "base editor"],
            "off_target_analysis": ["off-target effects", "CRISPR specificity", "genome-wide off-target"],
            "delivery_systems": ["CRISPR delivery", "viral vectors", "lipid nanoparticles", "electroporation"],
            "therapeutic_applications": ["CRISPR therapy", "gene therapy", "therapeutic genome editing"],
            "agricultural_applications": ["CRISPR agriculture", "crop improvement", "plant genome editing"],
            "model_organisms": ["CRISPR model organisms", "mouse models", "zebrafish", "drosophila"],
            "bioinformatics_tools": ["CRISPR bioinformatics", "design tools", "prediction algorithms"],
            "ethical_considerations": ["CRISPR ethics", "genome editing ethics", "regulatory considerations"],
            "clinical_trials": ["CRISPR clinical trials", "human trials", "safety assessment"]
        }

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("ComprehensiveRealMiner")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def mine_all_projects(self) -> Dict[str, List[Dict]]:
        """Mine data for all 11 projects using real APIs."""
        self.logger.info("Starting comprehensive mining for all 11 projects...")

        all_results = {}

        for project_name, queries in self.project_queries.items():
            self.logger.info(f"Mining data for project: {project_name}")
            project_results = []

            for query in queries:
                self.logger.info(f"Processing query: {query}")

                # Mine from all available sources
                tasks = [
                    self.encode_miner.mine_genomic_data(query),
                    self.ucsc_miner.mine_genome_annotations(),
                    self.ensembl_miner.mine_gene_annotations(),
                    self.geo_miner.mine_expression_data(query),
                    self.uniprot_miner.mine_protein_annotations(query),
                    self.pdb_miner.mine_structure_data(query),
                    self.clinical_miner.mine_clinical_trials(query)
                ]

                # Execute all mining tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Error in mining task {i}: {result}")
                    elif result:
                        project_results.extend(result)

                # Rate limiting between queries
                await asyncio.sleep(1)

            all_results[project_name] = project_results
            self.logger.info(f"Completed mining for {project_name}: {len(project_results)} records")

        return all_results

    async def save_to_bronze_layer(self, data: Dict[str, List[Dict]], output_dir: str = "data/bronze"):
        """Save mined data to Bronze layer in structured format."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for project_name, project_data in data.items():
            if project_data:
                # Save project data
                filename = f"{output_dir}/{project_name}_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(project_data, f, indent=2, default=str)

                self.logger.info(f"Saved {len(project_data)} records to {filename}")

                # Save summary statistics
                summary = {
                    "project": project_name,
                    "total_records": len(project_data),
                    "sources": list(set(record.get("source", "unknown") for record in project_data)),
                    "mined_at": timestamp,
                    "record_types": {}
                }

                for record in project_data:
                    source = record.get("source", "unknown")
                    summary["record_types"][source] = summary["record_types"].get(source, 0) + 1

                summary_filename = f"{output_dir}/{project_name}_{timestamp}_summary.json"
                with open(summary_filename, 'w') as f:
                    json.dump(summary, f, indent=2)

    async def run_comprehensive_mining(self):
        """Run the complete comprehensive mining pipeline."""
        try:
            self.logger.info("Starting comprehensive real data mining pipeline...")

            # Mine data for all projects
            all_data = await self.mine_all_projects()

            # Save to Bronze layer
            await self.save_to_bronze_layer(all_data)

            # Generate mining report
            total_records = sum(len(data) for data in all_data.values())
            self.logger.info(f"Comprehensive mining completed: {total_records} total records across all projects")

            return all_data

        except Exception as e:
            self.logger.error(f"Error in comprehensive mining: {e}")
            raise


# Main execution function
async def main():
    """Main function to run the comprehensive real mining pipeline."""
    miner = ComprehensiveRealMiner()
    await miner.run_comprehensive_mining()


if __name__ == "__main__":
    asyncio.run(main())
