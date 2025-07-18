"""
GeneX Mining System - Main Entry Point
Real-time data mining from scientific APIs
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import json
from src.mining.mvdp_validator import MVDPValidator
from src.api_clients.pubmed_client import PubMedClient
from src.api_clients.ensembl_client import EnsemblClient
from src.api_clients.crossref_client import CrossRefClient
from src.utils.config import Config
import aiohttp

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.mining.mining_engine import MiningEngine

# Configure structured JSON logging
logfile = "results/logs/pipeline.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("GeneXPipeline")

class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage()
        }
        return json.dumps(log_entry)

# Ensure a FileHandler is attached and set formatter
if not logger.handlers:
    handler = logging.FileHandler(logfile)
    logger.addHandler(handler)
if logger.handlers:
    logger.handlers[0].setFormatter(JsonLogFormatter())

async def mine_real_data(domain):
    """Mine real data for the specified domain"""
    config = Config()

    # Create HTTP session for all clients
    async with aiohttp.ClientSession() as session:
        # Initialize API clients with correct config structure and session
        pubmed_client = PubMedClient(config.get_api_config('pubmed'), session)
        ensembl_client = EnsemblClient(config.get_api_config('ensembl'), session)
        crossref_client = CrossRefClient(config.get_api_config('crossref'), session)

        # Domain-specific search terms
        domain_queries = {
            "CRISPR": [
                "CRISPR-Cas9 gene editing",
                "CRISPR therapy clinical trial",
                "Cas9 therapeutic applications",
                "CRISPR genome editing efficiency"
            ],
            "Prime": [
                "prime editing technology",
                "pegRNA prime editor",
                "PE3 prime editing",
                "prime editing therapeutic"
            ],
            "Base": [
                "base editing CRISPR",
                "cytosine base editor",
                "adenine base editor",
                "BE3 base editing"
            ]
        }

        all_papers = []
        queries = domain_queries.get(domain, [])

        for query in queries:
            try:
                logger.info(json.dumps({"event": f"search_pubmed_{domain}", "query": query}))

                # Search PubMed
                search_response = await pubmed_client.search_papers(
                    query=query,
                    max_results=50,  # Start with reasonable number
                    date_from="2020-01-01"  # Recent papers
                )

                if not search_response.success:
                    logger.error(json.dumps({
                        "event": f"pubmed_search_failed_{domain}",
                        "query": query,
                        "error": search_response.error_message
                    }))
                    continue

                # Parse search results to get PMIDs
                pmids = pubmed_client.parse_search_results(search_response)

                logger.info(json.dumps({
                    "event": f"pubmed_results_{domain}",
                    "query": query,
                    "papers_found": len(pmids)
                }))

                # Process each paper
                for pmid in pmids[:10]:  # Limit to first 10 for testing
                    try:
                        # Get detailed paper info
                        paper_response = await pubmed_client.get_paper_details(pmid)

                        if not paper_response.success:
                            logger.warning(json.dumps({
                                "event": f"paper_fetch_failed_{domain}",
                                "pmid": pmid,
                                "error": paper_response.error_message
                            }))
                            continue

                        # Parse paper details
                        papers = pubmed_client.parse_paper_details(paper_response)

                        if papers:
                            paper = papers[0]  # Take first paper

                            # Extract MVDP fields
                            mvdp_record = {
                                "persistent_id": paper.get('doi') or f"PMID:{pmid}",
                                "source_metadata": {
                                    "authors": paper.get('authors', []),
                                    "journal": paper.get('journal'),
                                    "year": paper.get('publication_date', '')[:4] if paper.get('publication_date') else None,
                                    "title": paper.get('title')
                                },
                                "organism_metadata": {
                                    "name": "Homo sapiens",  # Default, will be extracted from text
                                    "cell_line": None
                                },
                                "editing_technique": domain,
                                "target_gene": {
                                    "symbol": None,  # Will be extracted from text
                                    "ensembl_id": None
                                },
                                "guide_rna_sequence": None,  # Will be extracted from text
                                "delivery_method": None,  # Will be extracted from text
                                "efficiency_metric": None,  # Will be extracted from text
                                "efficiency_assay": None,  # Will be extracted from text
                                "off_target_assessment": None,  # Will be extracted from text
                                "off_target_assay": None,  # Will be extracted from text
                                "off_target_results": None  # Will be extracted from text
                            }

                            all_papers.append(mvdp_record)

                    except Exception as e:
                        logger.error(json.dumps({
                            "event": f"paper_processing_error_{domain}",
                            "pmid": pmid,
                            "error": str(e)
                        }))

            except Exception as e:
                logger.error(json.dumps({
                    "event": f"search_error_{domain}",
                    "query": query,
                    "error": str(e)
                }))

        return all_papers

def process_domain(domain, output_file):
    """Process a domain and save validated records"""
    logger.info(json.dumps({"event": f"pipeline_start_{domain}"}))

    # Run async mining
    papers = asyncio.run(mine_real_data(domain))

    # Validate records
    validator = MVDPValidator()
    valid_count = 0

    with open(output_file, "w") as fout:
        for paper in papers:
            is_valid, compliance = validator.validate(paper)
            log_entry = {
                "event": f"validate_record_{domain}",
                "persistent_id": paper.get("persistent_id"),
                "mvdprate": sum(compliance.values()) / len(compliance),
                "compliance": compliance
            }
            if is_valid:
                valid_count += 1
                fout.write(json.dumps(paper) + "\n")
            logger.info(json.dumps(log_entry))

    logger.info(json.dumps({
        "event": f"pipeline_end_{domain}",
        "valid_count": valid_count,
        "total": len(papers)
    }))

def main():
    """Main function to process all domains"""
    process_domain("CRISPR", "results/crispr_validated.jsonl")
    process_domain("Prime", "results/prime_validated.jsonl")
    process_domain("Base", "results/base_validated.jsonl")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Run the mining operation
    main()
