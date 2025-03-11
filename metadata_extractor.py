"""
Metadata Extractor for PDF documents.
Extracts metadata such as titles, dates, parties, and key entities from legal documents.
"""

import re
import logging
import spacy
from typing import Dict, Any, List, Tuple, Optional
import datetime
from sambanova_api import get_llama_response
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to load spaCy model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except (ImportError, OSError):
    logger.warning("spaCy model not found. Entity extraction will be limited.")
    HAS_SPACY = False

class MetadataExtractor:
    """Class to extract metadata from PDF documents, particularly legal documents."""
    
    def __init__(self):
        self.document_types = [
            "agreement", "contract", "amendment", "addendum", "memorandum", 
            "certificate", "letter", "notice", "policy", "deed", "declaration",
            "nda", "service agreement", "distribution agreement", "employment contract",
            "license agreement", "non-disclosure agreement", "master agreement"
        ]
    
    def extract_metadata(self, text: str, filename: str = None) -> Dict[str, Any]:
        """
        Extract metadata from document text, using LLM-based approach with fallback to rule-based.
        
        Args:
            text: The text content of the document
            filename: Optional filename for additional context
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            # Try LLM-based approach first
            metadata = self.extract_metadata_with_llm(text, filename)
            
            # Add estimated page count if not present
            if "estimated_page_count" not in metadata:
                metadata["estimated_page_count"] = self._estimate_page_count(text)
            
            # Standardize fields for consistency
            metadata = self._standardize_metadata_fields(metadata)
            
            return metadata
        except Exception as e:
            logger.error(f"Error in LLM metadata extraction: {str(e)}")
            
            # Fall back to rule-based extraction
            metadata = self._extract_metadata_rule_based(text, filename)
            
            # Standardize fields
            metadata = self._standardize_metadata_fields(metadata)
            
            return metadata
    
    def extract_metadata_with_llm(self, text: str, filename: str = None) -> Dict[str, Any]:
        """
        Extract metadata from text using LLM with robust error handling.
        
        Args:
            text: Text to extract metadata from
            filename: Optional filename to provide context
            
        Returns:
            Dictionary of metadata fields
        """
        try:
            logger.info("Extracting metadata with LLM")
            
            # Create comprehensive metadata extraction prompt
            prompt = (
                "Extract metadata from the following legal document text as a JSON object. Use the field guidelines below. "
                "Include each field only if it is found in the text.\n\n"
                "==== METADATA EXTRACTION GUIDELINES ====\n"
                "Include the following fields **only if they appear** in the text:\n"
                "- contract_type: The type of legal document (e.g., NDA, Service Agreement, Distribution Agreement, Employment Contract, License Agreement).\n"
                "- parties: The organizations or individuals involved.\n"
                "- effective_date: The date when the agreement takes effect (YYYY-MM-DD).\n"
                "- execution_date: The date when the document was signed.\n"
                "- termination_date: The date when the agreement ends or 'Indefinite'.\n"
                "- jurisdiction: The governing jurisdiction (e.g., France, EU, USA, Romanian Law).\n"
                "- governing_law: The legal framework (e.g., French Law, EU Regulations, Romanian Civil Code, GDPR Compliance).\n"
                "- version: The contract version or amendment indicator (e.g., V1, V2, Final, Draft).\n\n"
                
                "Additional Metadata (if present):\n"
                "- contract_status: Current status (Active, Expired, Terminated, Under Negotiation).\n"
                "- previous_version_reference: Reference to prior version(s) or version history.\n"
                "- key_obligations: Main responsibilities of the parties.\n"
                "- payment_obligations: Payment terms or financial obligations.\n"
                "- confidentiality_clause: Details of confidentiality or data protection obligations.\n"
                "- dispute_resolution: Mechanisms for resolving disputes (arbitration, litigation, etc.).\n"
                "- force_majeure: Conditions excusing performance (e.g., war, pandemic, government intervention).\n"
                "- exclusivity: Whether one party has exclusive rights.\n"
                "- non_compete: Restrictions on engaging with competitors.\n"
                "- ip_assignment: Ownership rights or licensing.\n\n"
                
                "==== FILENAME & DATE DETECTION ====\n"
                f"- The document filename is: {filename if filename else 'Unknown'}\n"
                "- If the text references any filename(s) containing a date (e.g., 'Contract_2023-01-05_v2.pdf'), parse that date.\n"
                "- Store any detected filename(s) under 'source_document' with the parsed date and a flag indicating whether it is the latest version.\n\n"
                
                "==== DOCUMENT TEXT ====\n"
                f"{text[:min(len(text), 8000)]}...\n\n"
                
                "Return ONLY a valid JSON object with the fields that are found. Omit any field that is not present. "
                "Do not add extra commentary or disclaimers."
            )
            
            # Get metadata from LLM
            llm_response = get_llama_response(prompt, temperature=0.1, max_tokens=1000)
            
            # Log the response
            logger.info("Received metadata extraction response from LLM")
            
            # Process the response
            if not llm_response:
                logger.error("LLM returned None response")
                return self._extract_metadata_rule_based(text, filename)
            
            # Try to parse JSON
            try:
                # Extract JSON part (in case there's additional text)
                json_match = re.search(r'({[\s\S]*})', llm_response)
                if json_match:
                    json_str = json_match.group(1)
                    metadata = json.loads(json_str)
                    return metadata
                else:
                    # No JSON found
                    logger.error("No JSON found in LLM response")
                    return self._extract_metadata_rule_based(text, filename)
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from LLM response: {e}")
                return self._extract_metadata_rule_based(text, filename)
                
        except Exception as e:
            logger.error(f"Error in metadata extraction: {str(e)}")
            return self._extract_metadata_rule_based(text, filename)
    
    def _extract_metadata_rule_based(self, text: str, filename: str = None) -> Dict[str, Any]:
        """
        Extract metadata using rule-based methods as a fallback.
        
        Args:
            text: Text to extract metadata from
            filename: Optional filename
            
        Returns:
            Dictionary of metadata fields
        """
        logger.info("Using rule-based metadata extraction")
        
        # Initialize metadata dict with basic structure
        result = {
            "title": "",
            "document_type": "",
            "estimated_page_count": self._estimate_page_count(text),
            "organizations": [],
            "people": [],
            "dates": [],
            "monetary_values": [],
            "contract_type": "",
            "source_document": {},
            "parties": [],
            "effective_date": "",
            "execution_date": "",
            "termination_date": "",
            "jurisdiction": "",
            "governing_law": "",
            "version": "",
            "contract_status": "",
            "exclusivity": "",
            "non_compete": "",
            "dispute_resolution": ""
        }
        
        # Add source document info if filename is provided
        if filename:
            result["source_document"] = self._extract_source_document_info(filename, text)
        
        # Extract title - check first 1000 characters
        title_text = text[:1000]
        title_match = re.search(r'^(.+?)(?:\n\n|\r\n\r\n)', title_text, re.DOTALL)
        if title_match:
            result["title"] = title_match.group(1).strip()
        else:
            # Try alternative patterns for title
            alt_title_match = re.search(r'(?:AGREEMENT|CONTRACT|MEMORANDUM|AMENDMENT)(?:\s+OF|\s+FOR|\s+TO)?\s+(.+?)(?:\n|\r\n)', title_text, re.IGNORECASE)
            if alt_title_match:
                result["title"] = alt_title_match.group(0).strip()
        
        # Extract document type
        for doc_type in self.document_types:
            if re.search(fr'\b{re.escape(doc_type)}\b', text[:3000], re.IGNORECASE):
                result["document_type"] = doc_type.title()
                result["contract_type"] = doc_type.title()
                break
        
        # Extract parties/organizations using pattern matching
        party_patterns = [
            r'(?:between|by and between)\s+(.+?)\s+and\s+(.+?)(?:\s+and\s+(.+?))?(?:\,|\.|;|\n)',
            r'(?:THIS\s+[A-Z]+\s+is made by|entered into by)\s+(.+?)\s+and\s+(.+?)(?:\s+and\s+(.+?))?(?:\,|\.|;|\n)',
            r'([A-Z][A-Za-z\s,]+(?:Inc\.|LLC|Ltd\.|Corporation|Corp\.|Co\.))\s+and\s+([A-Z][A-Za-z\s,]+(?:Inc\.|LLC|Ltd\.|Corporation|Corp\.|Co\.))'
        ]
        
        for pattern in party_patterns:
            parties_match = re.search(pattern, text[:5000], re.IGNORECASE | re.DOTALL)
            if parties_match:
                # Add all captured groups that aren't None
                parties = [group.strip() for group in parties_match.groups() if group]
                for party in parties:
                    # Clean up the party name
                    party = re.sub(r'\s+', ' ', party)
                    party = re.sub(r'[\(\"\'].*?[\)\"\']', '', party)  # Remove parenthetical text
                    party = party.strip('., \t\n\r')
                    
                    if party and len(party) > 3:  # Avoid very short strings
                        result["organizations"].append({"name": party, "type": "organization"})
                        result["parties"].append(party)
                break
        
        # Extract dates
        date_patterns = [
            # ISO format: YYYY-MM-DD
            r'\b(\d{4}-\d{2}-\d{2})\b',
            # Common US format: MM/DD/YYYY
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
            # Text format: Month DD, YYYY
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,\s+(\d{4})\b',
            # Short month format: MMM DD, YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?,\s+(\d{4})\b'
        ]
        
        for pattern in date_patterns:
            date_matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in date_matches:
                if pattern.startswith(r'\b(\d{4}-\d{2}-\d{2})'):
                    # ISO format
                    date_str = match.group(1)
                elif pattern.startswith(r'\b(\d{1,2}/\d{1,2}/\d{4})'):
                    # MM/DD/YYYY format - convert to ISO
                    parts = match.group(1).split('/')
                    date_str = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                else:
                    # Month name format - convert to ISO
                    month_name = match.group(1)
                    day = match.group(2)
                    year = match.group(3)
                    
                    month_dict = {
                        'january': '01', 'february': '02', 'march': '03', 'april': '04', 
                        'may': '05', 'june': '06', 'july': '07', 'august': '08', 
                        'september': '09', 'october': '10', 'november': '11', 'december': '12',
                        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 
                        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 
                        'oct': '10', 'nov': '11', 'dec': '12'
                    }
                    
                    month = month_dict.get(month_name.lower(), '01')
                    date_str = f"{year}-{month}-{day.zfill(2)}"
                
                if date_str not in result["dates"]:
                    result["dates"].append(date_str)
                
                # Check for effective date patterns
                effective_date_context = text[max(0, match.start() - 50):match.end() + 50]
                if re.search(r'effective\s+date|commencement\s+date|starts?\s+on|begins?\s+on', effective_date_context, re.IGNORECASE):
                    result["effective_date"] = date_str
                
                # Check for execution date patterns
                execution_date_context = text[max(0, match.start() - 50):match.end() + 50]
                if re.search(r'executed\s+on|signed\s+on|dated\s+as\s+of|as\s+of\s+the\s+date', execution_date_context, re.IGNORECASE):
                    result["execution_date"] = date_str
                
                # Check for termination date patterns
                termination_date_context = text[max(0, match.start() - 50):match.end() + 50]
                if re.search(r'terminat(es|ion)\s+on|expir(es|y|ation)\s+on|end(s|ing)\s+on|until', termination_date_context, re.IGNORECASE):
                    result["termination_date"] = date_str
        
        # Extract monetary values
        money_patterns = [
            r'\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(?:dollars|USD|€|EUR|£|GBP)',
            r'(?:USD|EUR|GBP)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in money_patterns:
            money_matches = re.finditer(pattern, text)
            for match in money_matches:
                value = match.group(1)
                if value not in result["monetary_values"]:
                    result["monetary_values"].append(value)
        
        # Extract jurisdiction and governing law
        law_patterns = [
            r'(?:governed by|subject to|pursuant to|in accordance with)[^\.\n]+((?:the laws of |the law of )?[A-Za-z\s]+(?:law|laws|jurisdiction))',
            r'jurisdiction[^\.\n]+(?:courts|tribunals)[^\.\n]+(?:of|in)\s+([A-Za-z\s]+)'
        ]
        
        for pattern in law_patterns:
            law_match = re.search(pattern, text, re.IGNORECASE)
            if law_match:
                law_text = law_match.group(1).strip()
                
                # Check if it specifies a jurisdiction or governing law
                if re.search(r'jurisdiction|venue|forum|courts of', law_match.group(0), re.IGNORECASE):
                    result["jurisdiction"] = law_text
                else:
                    result["governing_law"] = law_text
        
        # Extract contract status
        status_patterns = {
            "draft": r'\b(?:draft|proposal|for\s+review)\b',
            "executed": r'\b(?:executed|signed|finalized)\b',
            "expired": r'\b(?:expired|terminated|ended)\b',
            "active": r'\b(?:active|in\s+effect|in\s+force)\b'
        }
        
        for status, pattern in status_patterns.items():
            if re.search(pattern, text[:5000], re.IGNORECASE):
                result["contract_status"] = status.title()
                break
        
        # Extract exclusivity information
        exclusivity_match = re.search(r'(?:exclusiv(?:e|ity)|sole[ly]?)[^\.\n]+(?:right|distributor|provider|supplier)', text, re.IGNORECASE)
        if exclusivity_match:
            result["exclusivity"] = "Yes - " + exclusivity_match.group(0).strip()
        
        # Extract non-compete information
        non_compete_match = re.search(r'(?:non-compete|not\s+to\s+compete|shall\s+not\s+compete|refrain\s+from\s+competing)[^\.\n]+', text, re.IGNORECASE)
        if non_compete_match:
            result["non_compete"] = "Yes - " + non_compete_match.group(0).strip()
        
        # Extract dispute resolution
        dispute_patterns = [
            r'(?:dispute|controversy|claim)[^\.\n]+(?:shall|will|must)[^\.\n]+(?:arbitrat|mediat|courts\s+of)',
            r'(?:arbitration|mediation)[^\.\n]+(?:clause|provision|section)',
            r'(?:venue|forum|jurisdiction)[^\.\n]+(?:shall|will|must)[^\.\n]+(?:be|reside|exist)[^\.\n]+'
        ]
        
        for pattern in dispute_patterns:
            dispute_match = re.search(pattern, text, re.IGNORECASE)
            if dispute_match:
                result["dispute_resolution"] = dispute_match.group(0).strip()
                break
        
        # Add version information if filename contains version indicators
        if filename:
            version_match = re.search(r'v(\d+(?:\.\d+)?)|version\s+(\d+(?:\.\d+)?)', filename, re.IGNORECASE)
            if version_match:
                version = version_match.group(1) or version_match.group(2)
                result["version"] = f"V{version}"
        
        # Limit the number of entries for certain fields
        result["organizations"] = result["organizations"][:5]
        result["dates"] = result["dates"][:5]
        result["monetary_values"] = result["monetary_values"][:5]
        
        return result
    
    def _extract_source_document_info(self, filename: str, text: str = None) -> Dict[str, Any]:
        """Extract source document information from filename and text."""
        info = {
            "filename": filename,
            "parsed_date": None,
            "is_latest_version": True  # Default to True unless comparing multiple files
        }
        
        # Try to extract date from filename using various patterns
        date_patterns = [
            # YYYY-MM-DD or YYYY_MM_DD
            r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})',
            # DD-MM-YYYY or DD_MM_YYYY
            r'(\d{1,2})[-_](\d{1,2})[-_](\d{4})',
            # Simple year extraction
            r'[\D](\d{4})[\D]'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                if pattern.startswith(r'(\d{4})'):
                    # YYYY-MM-DD format
                    year = match.group(1)
                    month = match.group(2).zfill(2)
                    day = match.group(3).zfill(2)
                    info["parsed_date"] = f"{year}-{month}-{day}"
                    break
                elif pattern.startswith(r'(\d{1,2})[-_](\d{1,2})'):
                    # DD-MM-YYYY format
                    day = match.group(1).zfill(2)
                    month = match.group(2).zfill(2)
                    year = match.group(3)
                    info["parsed_date"] = f"{year}-{month}-{day}"
                    break
                elif pattern.startswith(r'[\D](\d{4})'):
                    # Just year
                    year = match.group(1)
                    info["parsed_date"] = f"{year}-01-01"  # Default to January 1st
                    break
        
        # Try to determine if this is a draft
        if re.search(r'draft|proposal', filename, re.IGNORECASE) or (text and re.search(r'DRAFT', text[:1000], re.IGNORECASE)):
            info["status"] = "Draft"
        else:
            info["status"] = "Final"
        
        return info
    
    def _estimate_page_count(self, text: str) -> int:
        """
        Estimate the number of pages based on text length.
        
        Args:
            text: Document text
            
        Returns:
            Estimated page count
        """
        # Rough estimate: ~3000 characters per page
        return max(1, len(text) // 3000)
    
    def _extract_dates(self, text: str) -> List[str]:
        """
        Extract dates mentioned in the document.
        
        Args:
            text: Document text
            
        Returns:
            List of date strings
        """
        # Patterns for common date formats
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            r'\b\d{4}-\d{1,2}-\d{1,2}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        # Filter out duplicates and sort
        unique_dates = list(set(dates))
        
        # Try to identify the most important date (usually near the beginning)
        effective_date = self._extract_effective_date(text)
        if effective_date and effective_date not in unique_dates:
            unique_dates.insert(0, effective_date)
        
        return unique_dates[:10]  # Limit to top 10 dates
    
    def _extract_effective_date(self, text: str) -> Optional[str]:
        """
        Extract the effective date of the document.
        
        Args:
            text: Document text
            
        Returns:
            Effective date string or None
        """
        effective_patterns = [
            r'(?:effective\s+(?:as\s+of\s+|date:?\s*))([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})',
            r'(?:dated\s+(?:as\s+of\s+)?)([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4})',
            r'(?:this\s+\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+)([A-Za-z]+,?\s+\d{4})'
        ]
        
        for pattern in effective_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return None
    
    def _extract_parties_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """
        Extract organizations and people using spaCy.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with organizations and people
        """
        # Process with spaCy (limit to first 10,000 chars for performance)
        doc = nlp(text[:10000])
        
        organizations = []
        people = []
        
        # Extract organizations and people
        for ent in doc.ents:
            if ent.label_ == "ORG" and len(ent.text) > 2:
                org = ent.text.strip()
                if org not in organizations:
                    organizations.append(org)
            elif ent.label_ == "PERSON" and len(ent.text) > 2:
                person = ent.text.strip()
                if person not in people:
                    people.append(person)
        
        # Also try pattern-based extraction for parties
        party_pattern = r'(?:between|among)([^.]+)(?:and)([^.]+)(?:\.|$)'
        party_match = re.search(party_pattern, text[:2000], re.IGNORECASE | re.DOTALL)
        
        if party_match:
            for i, party in enumerate([party_match.group(1), party_match.group(2)]):
                party = party.strip()
                # Check if it looks like an organization
                if any(term in party.lower() for term in ["inc", "corp", "llc", "ltd", "company", "corporation"]):
                    if party not in organizations:
                        organizations.append(party)
        
        return {
            "organizations": organizations[:10],  # Limit to top 10
            "people": people[:10]                # Limit to top 10
        }
    
    def _extract_parties_simple(self, text: str) -> Dict[str, List[str]]:
        """
        Simple pattern-based extraction for when spaCy is not available.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with organizations and people
        """
        organizations = []
        
        # Look for organization patterns
        org_patterns = [
            r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:Inc\.|LLC|Ltd\.|Corp\.|Corporation|Company)\b',
            r'([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\s+\((?:the\s+)?"(?:Company|Seller|Buyer|Vendor|Client|Contractor|Consultant|Employer|Employee)"\)',
            r'(?:between|among)\s+([^,]+),\s+(?:a|an)\s+([^,]+)'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    for submatch in match:
                        if submatch and len(submatch) > 3 and submatch not in organizations:
                            organizations.append(submatch.strip())
                elif match and len(match) > 3 and match not in organizations:
                    organizations.append(match.strip())
        
        # Simplistic approach for people - look for party references
        party_pattern = r'(?:between|among)([^.]+)(?:and)([^.]+)(?:\.|$)'
        party_match = re.search(party_pattern, text[:2000], re.IGNORECASE | re.DOTALL)
        
        people = []
        if party_match:
            parties = [party_match.group(1).strip(), party_match.group(2).strip()]
            for party in parties:
                # If it doesn't look like an organization, it might be a person
                if not any(term in party.lower() for term in ["inc", "corp", "llc", "ltd", "company", "corporation"]):
                    people.append(party)
        
        return {
            "organizations": organizations[:10],  # Limit to top 10
            "people": people[:5]                 # Limit to top 5
        }
    
    def _extract_monetary_values(self, text: str) -> List[str]:
        """
        Extract monetary values from the document.
        
        Args:
            text: Document text
            
        Returns:
            List of monetary value strings
        """
        # Patterns for monetary values
        money_patterns = [
            r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s+(?:million|billion|thousand))?',
            r'(?:USD|EUR|GBP|CAD|AUD)\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s+(?:dollars|euros|pounds)'
        ]
        
        monetary_values = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            monetary_values.extend(matches)
        
        # Filter out duplicates
        return list(set(monetary_values))[:10]  # Limit to top 10
    
    def _extract_all_dates(self, text: str) -> Dict[str, Any]:
        """
        Extract important dates including effective date, execution date, and termination date.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with date metadata
        """
        result = {}
        
        # Extract all dates as before
        all_dates = self._extract_dates(text)
        if all_dates:
            result["dates"] = all_dates
        
        # Extract effective date
        effective_date = self._extract_effective_date(text)
        if effective_date:
            # Try to standardize the date format to YYYY-MM-DD
            try:
                # Process different date formats
                if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', effective_date):
                    parts = effective_date.split('/')
                    if len(parts[2]) == 2:  # Handle two-digit year
                        year = int(parts[2])
                        year = 2000 + year if year < 50 else 1900 + year
                        parts[2] = str(year)
                    effective_date = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                elif re.match(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', effective_date, re.IGNORECASE):
                    date_obj = datetime.datetime.strptime(effective_date, "%B %d, %Y")
                    effective_date = date_obj.strftime("%Y-%m-%d")
                # Add more format conversions as needed
            except Exception as e:
                logger.warning(f"Failed to standardize date format: {e}")
            
            result["effective_date"] = effective_date
        
        # Extract execution date (when signed)
        execution_patterns = [
            r'(?:signed|executed|dated)(?:\s+as\s+of)?(?:\s+this)?(?:\s+on)?\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+)?(?:of\s+)?([A-Za-z]+)(?:,?\s+)(\d{4})',
            r'(?:signed|executed|dated)(?:\s+as\s+of)?(?:\s+this)?\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',
            r'(?:signed|executed|dated)(?:\s+as\s+of)?(?:\s+this)?\s+(\d{1,2}/\d{1,2}/\d{2,4})'
        ]
        
        for pattern in execution_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:  # Day, month, year format
                    day, month, year = match.groups()
                    try:
                        date_obj = datetime.datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
                        result["execution_date"] = date_obj.strftime("%Y-%m-%d")
                    except:
                        pass
                else:  # Other formats
                    execution_date = match.group(1)
                    try:
                        # Similar date standardization as effective date
                        # Implementation omitted for brevity
                        result["execution_date"] = execution_date
                    except:
                        pass
                break
        
        # Extract termination date
        termination_patterns = [
            r'(?:terminat(?:es|ion)\s+(?:on|date))(?:\s+the)?\s+(?:date\s+of\s+)?([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2})',
            r'(?:shall\s+remain\s+in\s+(?:full\s+)?(?:force\s+and\s+)?effect\s+until)(?:\s+the)?\s+(?:date\s+of\s+)?([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2})',
            r'(?:continue\s+in\s+effect\s+until)(?:\s+the)?\s+(?:date\s+of\s+)?([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2})'
        ]
        
        for pattern in termination_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["termination_date"] = match.group(1)
                break
                
        # Check for indefinite term
        indefinite_patterns = [
            r'(?:shall\s+remain\s+in\s+(?:full\s+)?(?:force\s+and\s+)?effect\s+until\s+terminated)',
            r'(?:continue\s+in\s+(?:full\s+)?(?:force\s+and\s+)?effect\s+until\s+terminated)',
            r'(?:no\s+fixed\s+term|indefinite\s+term|perpetual\s+term)'
        ]
        
        for pattern in indefinite_patterns:
            if re.search(pattern, text, re.IGNORECASE) and "termination_date" not in result:
                result["termination_date"] = "Indefinite"
                break
                
        return result
        
    def _extract_legal_framework(self, text: str) -> Dict[str, str]:
        """
        Extract information about jurisdiction and governing law.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with jurisdiction and governing law information
        """
        result = {}
        
        # Extract jurisdiction
        jurisdiction_patterns = [
            r'(?:jurisdiction|venue)\s+(?:of|in)\s+the\s+(?:courts\s+of|state\s+of|country\s+of)\s+([A-Za-z\s]+)',
            r'(?:exclusive\s+)?jurisdiction\s+(?:shall\s+be|is|will\s+be)\s+(?:the\s+)?(?:courts\s+of|state\s+of|country\s+of)\s+([A-Za-z\s]+)',
            r'disputes\s+(?:shall|will)\s+be\s+(?:resolved|settled)\s+in\s+(?:the\s+)?(?:courts\s+of|state\s+of|country\s+of)\s+([A-Za-z\s]+)'
        ]
        
        for pattern in jurisdiction_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                jurisdiction = match.group(1).strip()
                # Clean up common trailing words
                jurisdiction = re.sub(r'\s+and\s+.*$', '', jurisdiction)
                result["jurisdiction"] = jurisdiction
                break
                
        # Extract governing law
        law_patterns = [
            r'(?:governed|interpreted)\s+(?:by|in\s+accordance\s+with)\s+the\s+laws\s+of\s+(?:the\s+)?([A-Za-z\s]+)',
            r'governing\s+law\s+(?:shall\s+be|is|will\s+be)\s+the\s+laws?\s+of\s+(?:the\s+)?([A-Za-z\s]+)',
            r'this\s+agreement\s+(?:shall\s+be|is|will\s+be)\s+(?:governed|interpreted)\s+(?:by|in\s+accordance\s+with)\s+([A-Za-z\s]+)\s+law'
        ]
        
        for pattern in law_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                law = match.group(1).strip()
                # Clean up common trailing words
                law = re.sub(r'\s+and\s+.*$', '', law)
                result["governing_law"] = law
                break
                
        return result
        
    def _extract_contract_status(self, text: str) -> Dict[str, Any]:
        """
        Extract information about contract status, version, and related metadata.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with contract status and version information
        """
        result = {}
        
        # Extract version
        version_patterns = [
            r'(?:version|ver|v)[\.:\s]+(\d+(?:\.\d+)?)',
            r'(?:revision|rev)[\.:\s]+(\d+(?:\.\d+)?)',
            r'\b(draft|final)\b',
            r'amendment\s+(?:no\.|number|#)?\s*(\d+)'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)  # Look in the first part of document
            if match:
                result["version"] = match.group(1).upper() if match.group(1).lower() == "draft" or match.group(1).lower() == "final" else match.group(1)
                break
                
        # Extract contract status
        status_patterns = [
            r'\b(active|executed|signed|in\s+effect)\b',
            r'\b(draft|under\s+negotiation|pending\s+signature)\b',
            r'\b(expired|terminated|canceled|cancelled)\b'
        ]
        
        status_mapping = {
            'active': 'Active', 'executed': 'Active', 'signed': 'Active', 'in effect': 'Active',
            'draft': 'Under Negotiation', 'under negotiation': 'Under Negotiation', 'pending signature': 'Under Negotiation',
            'expired': 'Expired', 'terminated': 'Terminated', 'canceled': 'Terminated', 'cancelled': 'Terminated'
        }
        
        for pattern in status_patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                status_key = match.group(1).lower()
                result["contract_status"] = status_mapping.get(status_key, status_key.title())
                break
                
        # Extract previous version reference
        prev_version_patterns = [
            r'(?:previous|prior)\s+(?:version|agreement)(?:\s+dated)?\s+([A-Za-z0-9\s,\.]+)',
            r'(?:supersedes|replaces)\s+(?:the|that\s+certain)?\s+([A-Za-z0-9\s,\.]+)(?:\s+dated\s+[A-Za-z0-9\s,\.]+)?',
            r'amendment\s+(?:to|of)\s+(?:the|that\s+certain)?\s+([A-Za-z0-9\s,\.]+)(?:\s+dated\s+[A-Za-z0-9\s,\.]+)?'
        ]
        
        for pattern in prev_version_patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                result["previous_version_reference"] = match.group(1).strip()
                break
                
        return result
        
    def _extract_legal_clauses(self, text: str) -> Dict[str, Any]:
        """
        Extract key legal clauses and their details.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with legal clause information
        """
        result = {}
        
        # Extract confidentiality clause
        confidentiality_patterns = [
            r'(?:confidentiality|non[-\s]*disclosure)(?:[^.]{0,100}?)(?:clause|provision|section|agreement)(?:[^.]{10,500}?)(?:\.)',
            r'(?:section|clause|article)[^.]{0,50}confidential[^.]{10,500}?(?:\.)'
        ]
        
        for pattern in confidentiality_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["confidentiality_clause"] = match.group(0).strip()
                # Check for confidentiality duration
                duration_match = re.search(r'(?:period\s+of|for\s+a\s+period\s+of|duration\s+of)\s+(\d+)\s+(years|months)', result["confidentiality_clause"], re.IGNORECASE)
                if duration_match:
                    result["confidentiality_duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"
                break
        
        # Extract dispute resolution
        dispute_patterns = [
            r'(?:dispute\s+resolution|arbitration|mediation|litigation)(?:[^.]{0,100}?)(?:clause|provision|section)(?:[^.]{10,300}?)(?:\.)',
            r'(?:disputes|claims|controversies)[^.]{0,100}?(?:shall|will|must)[^.]{0,100}?(?:be\s+resolved|be\s+settled|be\s+determined)[^.]{10,300}?(?:\.)'
        ]
        
        for pattern in dispute_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["dispute_resolution"] = match.group(0).strip()
                break
        
        # Extract force majeure
        force_majeure_patterns = [
            r'(?:force\s+majeure)(?:[^.]{0,100}?)(?:clause|provision|section)(?:[^.]{10,500}?)(?:\.)',
            r'(?:events?\s+beyond[^.]{0,50}?control|act\s+of\s+god)[^.]{10,500}?(?:\.)'
        ]
        
        for pattern in force_majeure_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["force_majeure"] = match.group(0).strip()
                break
        
        # Extract exclusivity
        exclusivity_patterns = [
            r'(?:exclusiv(?:e|ity)|sole\s+(?:provider|supplier|distributor))(?:[^.]{0,100}?)(?:clause|provision|section|rights?)(?:[^.]{10,300}?)(?:\.)',
            r'(?:exclusive\s+right|sole\s+right)[^.]{10,300}?(?:\.)'
        ]
        
        for pattern in exclusivity_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["exclusivity"] = match.group(0).strip()
                
                # Also extract exclusivity agreement details
                if "exclusivity_agreement" not in result:
                    exclusivity_details = re.search(r'(?:exclusive|sole)[^.]{0,30}?(?:in|for|within|throughout)[^.]{0,100}?([^.]{0,200}?)(?:\.)', match.group(0), re.IGNORECASE | re.DOTALL)
                    if exclusivity_details:
                        result["exclusivity_agreement"] = exclusivity_details.group(0).strip()
                
                break
        
        # Extract non-compete
        non_compete_patterns = [
            r'(?:non[-\s]*compete|competition\s+restrictions?)(?:[^.]{0,100}?)(?:clause|provision|section|agreement)(?:[^.]{10,500}?)(?:\.)',
            r'(?:shall|will|must)\s+not[^.]{0,100}?(?:compete|competing|competition)[^.]{10,300}?(?:\.)'
        ]
        
        for pattern in non_compete_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["non_compete"] = match.group(0).strip()
                
                # Also extract non-compete duration
                duration_match = re.search(r'(?:period\s+of|for\s+a\s+period\s+of|duration\s+of)\s+(\d+)\s+(years|months)', match.group(0), re.IGNORECASE)
                if duration_match:
                    result["post_contract_restriction_period"] = f"{duration_match.group(1)} {duration_match.group(2)}"
                
                break
        
        # Extract termination details
        termination_patterns = [
            r'(?:termination|cancellation)(?:[^.]{0,100}?)(?:clause|provision|section)(?:[^.]{10,500}?)(?:\.)',
            r'(?:this\s+agreement\s+may\s+be\s+terminated|either\s+party\s+may\s+terminate)[^.]{10,500}?(?:\.)'
        ]
        
        for pattern in termination_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                termination_text = match.group(0).strip()
                
                # Extract termination notice period
                notice_match = re.search(r'(?:notice\s+(?:of|period\s+of)|written\s+notice)[^.]{0,20}?(\d+)[^.]{0,20}?(days|months)', termination_text, re.IGNORECASE)
                if notice_match:
                    result["termination_notice_period"] = f"{notice_match.group(1)} {notice_match.group(2)}"
                
                # Extract grounds for termination
                grounds_match = re.search(r'(?:terminate[^.]{0,50}?(?:if|in\s+the\s+event|upon|for)[^.]{10,200}?(?:\.|;))', termination_text, re.IGNORECASE | re.DOTALL)
                if grounds_match:
                    result["grounds_for_termination"] = grounds_match.group(0).strip()
                
                # Check for automatic renewal
                if re.search(r'(?:automatically\s+renew|automatic\s+renewal|renew\s+automatically)', termination_text, re.IGNORECASE):
                    result["automatic_renewal"] = True
                
                break
                
        # Extract IP assignment
        ip_patterns = [
            r'(?:intellectual\s+property|IP)(?:[^.]{0,100}?)(?:assignment|ownership|rights)(?:[^.]{10,500}?)(?:\.)',
            r'(?:ownership\s+of|title\s+to)[^.]{0,50}?(?:intellectual\s+property|patents|trademarks|copyrights)[^.]{10,300}?(?:\.)'
        ]
        
        for pattern in ip_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["ip_assignment"] = match.group(0).strip()
                break
                
        return result
        
    def _extract_business_terms(self, text: str) -> Dict[str, Any]:
        """
        Extract business-specific terms and obligations.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with business term information
        """
        result = {}
        
        # Extract key obligations
        obligation_patterns = [
            r'(?:obligations|responsibilities|duties)[^.]{0,50}?(?:of\s+the\s+parties|of\s+(?:buyer|seller|licensor|licensee|customer|vendor|supplier))[^.]{10,500}?(?:\.)',
            r'(?:buyer|seller|licensor|licensee|customer|vendor|supplier)[^.]{0,20}?(?:shall|will|must|agrees\s+to)[^.]{10,300}?(?:\.)'
        ]
        
        key_obligations = []
        for pattern in obligation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches[:3]:  # Limit to 3 key obligations
                if isinstance(match, tuple):
                    match = match[0]  # In case the regex returns groups
                if match and len(match.strip()) > 20:  # Ensure it's substantial
                    key_obligations.append(match.strip())
        
        if key_obligations:
            result["key_obligations"] = key_obligations
        
        # Extract payment obligations
        payment_patterns = [
            r'(?:payment\s+terms|payment\s+obligations|fees|pricing)[^.]{0,100}?(?:clause|provision|section)(?:[^.]{10,500}?)(?:\.)',
            r'(?:buyer|customer|licensee)[^.]{0,30}?(?:shall|will|must|agrees\s+to)[^.]{0,50}?(?:pay|remit|transfer)[^.]{10,300}?(?:\.)'
        ]
        
        for pattern in payment_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["payment_obligations"] = match.group(0).strip()
                
                # Extract royalty fee percentage if present
                royalty_match = re.search(r'(?:royalty|fee)[^.]{0,30}?(\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s+percent)', match.group(0), re.IGNORECASE)
                if royalty_match:
                    result["royalty_fee_percentage"] = royalty_match.group(1).strip()
                
                # Extract late payment penalty if present
                penalty_match = re.search(r'(?:late\s+payment|overdue)[^.]{0,50}?(\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s+percent)', match.group(0), re.IGNORECASE)
                if penalty_match:
                    result["late_payment_penalty"] = penalty_match.group(1).strip()
                
                break
        
        # Extract trademark information
        trademark_patterns = [
            r'(?:trademark|brand|logo)[^.]{0,100}?(?:ownership|rights|license|usage)[^.]{10,500}?(?:\.)',
            r'(?:right\s+to\s+use|license\s+to\s+use)[^.]{0,50}?(?:trademark|brand|logo|name)[^.]{10,300}?(?:\.)'
        ]
        
        for pattern in trademark_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["trademark_ownership"] = match.group(0).strip()
                
                # Check for exclusive/non-exclusive
                if re.search(r'exclusive', match.group(0), re.IGNORECASE):
                    result["trademark_usage_license"] = "Exclusive"
                elif re.search(r'non[\s-]*exclusive', match.group(0), re.IGNORECASE):
                    result["trademark_usage_license"] = "Non-exclusive"
                
                break
        
        # Extract audit rights
        audit_patterns = [
            r'(?:audit|inspection)[^.]{0,100}?(?:rights|clause|provision|section)[^.]{10,300}?(?:\.)',
            r'(?:right\s+to\s+audit|right\s+to\s+inspect|may\s+audit|may\s+inspect)[^.]{10,300}?(?:\.)'
        ]
        
        for pattern in audit_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["audit_rights"] = match.group(0).strip()
                break
        
        # Extract data protection/privacy
        data_patterns = [
            r'(?:data\s+protection|privacy|GDPR|personal\s+data)[^.]{0,100}?(?:clause|provision|section|compliance)[^.]{10,500}?(?:\.)',
            r'(?:processing\s+of\s+personal\s+data|processing\s+personal\s+information)[^.]{10,500}?(?:\.)'
        ]
        
        for pattern in data_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["data_processing_agreement"] = match.group(0).strip()
                break
        
        # Extract marketing restrictions
        marketing_patterns = [
            r'(?:marketing|advertising|promotion)[^.]{0,100}?(?:restrictions|limitations|requirements|approval)[^.]{10,300}?(?:\.)',
            r'(?:prior\s+approval|prior\s+written\s+consent)[^.]{0,50}?(?:marketing|advertising|promotional)[^.]{10,300}?(?:\.)'
        ]
        
        for pattern in marketing_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result["marketing_approval_requirement"] = match.group(0).strip()
                break
                
        return result
        
    def _extract_source_document(self, text: str) -> Dict[str, Any]:
        """
        Extract information about the source document filename and date if present.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with source document information
        """
        result = {}
        
        # Look for filename patterns with dates
        filename_patterns = [
            r'(?:file|document|agreement)(?:[:\s]+)([A-Za-z0-9_-]+_\d{4}[-/]\d{1,2}[-/]\d{1,2}[_v\d\.]*\.(?:pdf|docx|doc))',
            r'([A-Za-z0-9_-]+_\d{4}[-/]\d{1,2}[-/]\d{1,2}[_v\d\.]*\.(?:pdf|docx|doc))',
            r'([A-Za-z0-9_-]+[_v\d\.]*_\d{4}[-/]\d{1,2}[-/]\d{1,2}\.(?:pdf|docx|doc))'
        ]
        
        found_filenames = []
        for pattern in filename_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_filenames.extend(matches)
        
        if found_filenames:
            # Process the first found filename or the one that seems most relevant
            filename = found_filenames[0]
            
            # Extract date from filename
            date_match = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', filename)
            if date_match:
                year, month, day = date_match.groups()
                parsed_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                
                # Create source document metadata
                result["source_document"] = {
                    "filename": filename,
                    "parsed_date": parsed_date,
                    "is_latest_version": True  # Default assumption unless multiple files are compared
                }
        
        return result

    def _extract_document_version(self, text: str, filename: str = None) -> Dict[str, Any]:
        """Extract document version information."""
        result = {}
        
        # First try to extract from filename if available
        if filename:
            # Look for patterns like v1.2, version 2, _v3, etc.
            version_pattern = re.search(r'[_-]v(\d+(?:\.\d+)?)|version\s+(\d+(?:\.\d+)?)', filename, re.IGNORECASE)
            if version_pattern:
                version = version_pattern.group(1) or version_pattern.group(2)
                result["version"] = version
                
            # Look for date patterns in filename (often indicates version)
            date_pattern = re.search(r'(\d{4}[-_]\d{1,2}[-_]\d{1,2}|\d{1,2}[-_]\d{1,2}[-_]\d{4})', filename)
            if date_pattern:
                result["version_date"] = date_pattern.group(1).replace('_', '-')
        
        # Then look in document text
        version_patterns = [
            r'version\s*[:=]?\s*(\d+(?:\.\d+)?)',
            r'document\s+version\s*[:=]?\s*(\d+(?:\.\d+)?)',
            r'revision\s*[:=]?\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                result["version"] = match.group(1)
                break
                
        return result

    def compare_document_versions(self, filenames: List[str]) -> Dict[str, Any]:
        """
        Compare multiple filenames to determine the latest version.
        
        Args:
            filenames: List of filenames to compare
            
        Returns:
            Dictionary with latest version information
        """
        result = {"latest_version": None, "latest_date": None}
        
        latest_date = None
        latest_file = None
        
        # Extract dates from filenames
        for filename in filenames:
            # Try different date formats in filenames
            date_matches = re.findall(r'(\d{2}|\d{4})[-_](\d{1,2})[-_](\d{1,2})|(\d{1,2})[-_](\d{1,2})[-_](\d{2}|\d{4})', filename)
            
            if date_matches:
                match = date_matches[0]
                
                # Handle both date formats: year-month-day or day-month-year
                if match[0]:  # year-month-day format
                    year = match[0] if len(match[0]) == 4 else f"20{match[0]}"  # Assume 20xx for 2-digit years
                    month = match[1]
                    day = match[2]
                else:  # day-month-year format
                    day = match[3]
                    month = match[4]
                    year = match[5] if len(match[5]) == 4 else f"20{match[5]}"
                
                try:
                    # Create date object for comparison
                    date_obj = datetime.datetime(int(year), int(month), int(day))
                    
                    # If this is the latest date so far, update
                    if latest_date is None or date_obj > latest_date:
                        latest_date = date_obj
                        latest_file = filename
                        
                        # Format date as string
                        result["latest_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        result["latest_version"] = filename
                except ValueError:
                    # Invalid date values
                    continue
        
        return result

    def _standardize_metadata_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize metadata fields to ensure consistency between rule-based and LLM-based extraction.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Standardized metadata dictionary
        """
        # Create a new dictionary for standardized fields
        standardized = metadata.copy()
        
        # Map common field variations
        field_mappings = {
            # LLM field name → Standard field name
            "contract_type": "document_type",
            "parties": "organizations",
            "effective_date": "dates",
            "execution_date": "dates",
            "termination_date": "dates"
        }
        
        # Apply mappings
        for llm_field, standard_field in field_mappings.items():
            if llm_field in metadata and metadata[llm_field]:
                # If the field exists in metadata and has a value
                if standard_field not in standardized:
                    standardized[standard_field] = []
                    
                # Handle different field types
                if standard_field == "dates":
                    # For dates, add to the dates array
                    if isinstance(metadata[llm_field], list):
                        standardized[standard_field].extend(metadata[llm_field])
                    else:
                        standardized[standard_field].append(metadata[llm_field])
                elif standard_field == "organizations":
                    # For organizations, add to the organizations array
                    if isinstance(metadata[llm_field], list):
                        standardized[standard_field].extend(metadata[llm_field])
                    else:
                        standardized[standard_field].append(metadata[llm_field])
                else:
                    # For other fields, just copy the value
                    standardized[standard_field] = metadata[llm_field]
        
        # Ensure common fields exist with default values
        if "document_type" not in standardized:
            standardized["document_type"] = "Unknown"
        if "organizations" not in standardized:
            standardized["organizations"] = []
        if "dates" not in standardized:
            standardized["dates"] = []
        if "monetary_values" not in standardized:
            standardized["monetary_values"] = []
        
        return standardized 