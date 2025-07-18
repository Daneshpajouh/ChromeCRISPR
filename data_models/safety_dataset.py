#!/usr/bin/env python3
"""
Safety Dataset Data Model
Captures safety, toxicity, off-target, and adverse event features for gene editing experiments
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class OffTargetEvent:
    sequence: str
    score: float
    position: str
    gene: Optional[str] = None
    effect: Optional[str] = None
    assay: Optional[str] = None
    frequency: Optional[float] = None
    severity: Optional[str] = None
    mitigation: Optional[List[str]] = None

@dataclass
class AdverseEvent:
    event_type: str  # e.g., cytotoxicity, apoptosis, immune response
    value: float
    unit: Optional[str] = None
    description: Optional[str] = None
    time_point: Optional[str] = None
    severity: Optional[str] = None

@dataclass
class SafetyRecord:
    persistent_id: str
    editing_technique: str
    cell_line: str
    organism: str
    tissue_type: Optional[str] = None
    delivery_method: Optional[str] = None
    protocol: Optional[str] = None

    # Safety metrics
    cytotoxicity: Optional[float] = None
    cell_viability: Optional[float] = None
    apoptosis_rate: Optional[float] = None
    necrosis_rate: Optional[float] = None
    cell_proliferation: Optional[float] = None
    dna_damage: Optional[float] = None
    oxidative_stress: Optional[float] = None
    inflammatory_response: Optional[float] = None
    immune_response: Optional[float] = None
    genotoxicity: Optional[float] = None
    teratogenicity: Optional[float] = None
    carcinogenicity: Optional[float] = None

    # Off-target analysis
    off_target_events: List[OffTargetEvent] = field(default_factory=list)

    # Adverse events
    adverse_events: List[AdverseEvent] = field(default_factory=list)

    # Overall safety score
    safety_score: Optional[float] = None

    # Source and provenance
    source_metadata: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'persistent_id': self.persistent_id,
            'editing_technique': self.editing_technique,
            'cell_line': self.cell_line,
            'organism': self.organism,
            'tissue_type': self.tissue_type,
            'delivery_method': self.delivery_method,
            'protocol': self.protocol,
            'cytotoxicity': self.cytotoxicity,
            'cell_viability': self.cell_viability,
            'apoptosis_rate': self.apoptosis_rate,
            'necrosis_rate': self.necrosis_rate,
            'cell_proliferation': self.cell_proliferation,
            'dna_damage': self.dna_damage,
            'oxidative_stress': self.oxidative_stress,
            'inflammatory_response': self.inflammatory_response,
            'immune_response': self.immune_response,
            'genotoxicity': self.genotoxicity,
            'teratogenicity': self.teratogenicity,
            'carcinogenicity': self.carcinogenicity,
            'off_target_events': [
                {
                    'sequence': event.sequence,
                    'score': event.score,
                    'position': event.position,
                    'gene': event.gene,
                    'effect': event.effect,
                    'assay': event.assay,
                    'frequency': event.frequency,
                    'severity': event.severity,
                    'mitigation': event.mitigation
                } for event in self.off_target_events
            ],
            'adverse_events': [
                {
                    'event_type': event.event_type,
                    'value': event.value,
                    'unit': event.unit,
                    'description': event.description,
                    'time_point': event.time_point,
                    'severity': event.severity
                } for event in self.adverse_events
            ],
            'safety_score': self.safety_score,
            'source_metadata': self.source_metadata,
            'provenance': self.provenance,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)
