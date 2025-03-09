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
        Extract metadata from document text using LLM-based approach.
        
        Args:
            text: The text content of the document
            filename: Optional filename for additional context
            
        Returns:
            Dictionary containing extracted metadata in JSON format
        """
        # Add filename to context if available
        filename_context = f"Filename: {filename}\n\n" if filename else ""
        
        # Create the extraction prompt
        prompt = (
            "Extract metadata from the following legal document text as a JSON object. Use the field guidelines below. "
            "Include each field only if it is found in the text.\n\n"
            f"{filename_context}"
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
            "==== ADDITIONAL FIELDS FROM SPECIFIC LABELS ====\n"
            "Also include these fields if they appear in the text:\n"
            "- brand_licensor: Entity granting rights over brand use.\n"
            "- licensee: Entity receiving brand/IP rights.\n"
            "- producers: Companies authorized to manufacture products.\n"
            "- partner_restaurants: Entities authorized to sell branded products.\n"
            "- sub_licensee: Third parties allowed to use IP under a sublicense.\n"
            "- competitor_restriction: Details on non-compete clauses or market exclusivity.\n"
            "- trademark_ownership: Ownership of trademarks and licensing terms.\n"
            "- trademark_registration_status: Registration status of trademarks (e.g., Pending, Approved).\n"
            "- trademark_usage_license: Details on trademark licensing (Exclusive/Non-exclusive, Geographic Scope).\n"
            "- know_how_transfer: Clauses related to transferring know-how.\n"
            "- trade_secrets_protection: Confidentiality terms around proprietary knowledge.\n"
            "- branding_rights: Rights regarding logos, slogans, and product names.\n"
            "- advertising_restrictions: Approval requirements for marketing materials.\n"
            "- royalty_fee_percentage: Percentage payable as a royalty fee.\n"
            "- revenue_share_model: Details on revenue sharing between parties.\n"
            "- late_payment_penalty: Penalties or interest for late payments.\n"
            "- revenue_collection_agent: Entity authorized to collect payments.\n"
            "- obligation_to_perform: Specific performance obligations defined in the contract.\n"
            "- service_standards: Quality control and performance measures.\n"
            "- product_quality_standards: Health and safety compliance details.\n"
            "- compliance_requirements: Legal compliance obligations (e.g., GDPR, Consumer Protection Laws).\n"
            "- audit_rights: Rights to inspect compliance with contract terms.\n"
            "- penalties_for_breach: Consequences for non-performance.\n"
            "- termination_notice_period: Notice period required for termination.\n"
            "- automatic_renewal: Whether the contract renews automatically.\n"
            "- grounds_for_termination: Conditions that allow contract termination.\n"
            "- post_termination_restrictions: Obligations after termination (e.g., non-compete clauses).\n"
            "- survival_clauses: Clauses that remain in effect post-termination.\n"
            "- exit_compensation: Penalties or fees for early termination.\n"
            "- exclusivity_agreement: Details on any exclusivity agreements.\n"
            "- market_restrictions: Geographic or sector-specific limitations.\n"
            "- competitor_collaboration_ban: Restrictions on partnering with competitors.\n"
            "- post_contract_restriction_period: Duration of non-compete obligations after contract ends.\n"
            "- data_processing_agreement: GDPR compliance details related to data processing.\n"
            "- third_party_disclosure_restrictions: Limits on sharing confidential information with third parties.\n"
            "- confidentiality_duration: Duration for which confidentiality obligations persist.\n"
            "- sensitive_data_definition: Definitions regarding sensitive or proprietary data.\n"
            "- security_measures: Measures such as encryption or access control details.\n"
            "- marketing_approval_requirement: Whether marketing materials require prior approval.\n"
            "- co_branding_agreements: Permissions for joint branding initiatives.\n"
            "- use_of_trademark_in_ads: Allowed usage of trademarks in advertising.\n"
            "- sales_channel_limitations: Restrictions on sales channels (online vs. offline).\n"
            "- influencer_advertising_restrictions: Terms protecting brand image in influencer campaigns.\n"
            "- reporting_requirements: Requirements for sales or performance reporting.\n"
            "- kpi_tracking: Key performance indicators mentioned in the contract.\n"
            "- performance_bonuses: Bonus structures tied to performance metrics.\n"
            "- inspection_rights: Rights to conduct inspections or audits.\n"
            "==== FILENAME & DATE DETECTION ====\n"
            "- If the text references any filename(s) containing a date (e.g., 'Contract_2023-01-05_v2.pdf'), parse that date.\n"
            "- Compare multiple filenames to determine which is earliest or latest.\n"
            "- Store any detected filename(s) under 'source_document' with the parsed date and a flag indicating whether it is the latest version.\n\n"
            "==== DOCUMENT TEXT ====\n"
            f"Document Text (first section):\n{text[:3000]}...\n\n"
            "Return ONLY a valid JSON object with the fields that are found. Omit any field that is not present. "
            "Do not add extra commentary or disclaimers."
        )
        
        logger.info("Extracting metadata using LLM")
        response = get_llama_response(prompt, temperature=0.0, max_tokens=1500)
        
        try:
            # Try to parse the JSON response
            # Strip any possible markdown formatting if present
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response.strip()
            
            metadata = json.loads(json_str)
            
            # If the response has a nested metadata object, use that
            if isinstance(metadata, dict) and "metadata" in metadata:
                return metadata["metadata"]
            
            return metadata
        except Exception as e:
            logger.error(f"Error parsing LLM metadata response: {str(e)}")
            logger.info(f"Raw response: {response}")
            
            # Fall back to rule-based extraction
            logger.info("Falling back to rule-based extraction")
            return self._extract_metadata_rule_based(text, filename)
    
    def _extract_metadata_rule_based(self, text: str, filename: str = None) -> Dict[str, Any]:
        """Rule-based metadata extraction (fallback method)"""
        metadata = {}
        
        # Extract document type and title
        metadata.update(self._extract_document_type(text))
        
        # Extract dates
        metadata.update(self._extract_all_dates(text))
        
        # Extract parties
        if HAS_SPACY:
            metadata.update(self._extract_parties_with_spacy(text))
        else:
            metadata.update(self._extract_parties_simple(text))
        
        # Extract monetary values
        metadata["monetary_values"] = self._extract_monetary_values(text)
        
        # Extract legal jurisdiction and governing law
        metadata.update(self._extract_legal_framework(text))
        
        # Extract contract status, version and related metadata
        metadata.update(self._extract_contract_status(text))
        
        # Extract key legal clauses
        metadata.update(self._extract_legal_clauses(text))
        
        # Extract business terms
        metadata.update(self._extract_business_terms(text))
        
        # Extract filename and version information if present
        metadata.update(self._extract_source_document(text))
        
        # Add estimated page count
        metadata["estimated_page_count"] = self._estimate_page_count(text)
        
        # Check for version info if filename provided
        if filename:
            metadata.update(self._extract_document_version(text, filename))
        
        # Clean metadata by removing empty values
        metadata = {k: v for k, v in metadata.items() if v}
        
        return metadata
    
    def _extract_document_type(self, text: str) -> Dict[str, str]:
        """
        Extract document type and title.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with document type and title
        """
        # Extract first 1000 characters for examination
        header_text = text[:1000].lower()
        
        # Default values
        doc_type = "Unknown"
        title = "Untitled Document"
        
        # Check for document type keywords
        for type_keyword in self.document_types:
            if type_keyword in header_text:
                doc_type = type_keyword.title()
                
                # Try to extract the full title using regex
                title_pattern = re.compile(
                    rf"(?:this)?\s*{type_keyword}\s+(?:between|among|of|for|with|by)?\s*(.*?)(?:dated|made|entered|this\s+\d|\n\n)",
                    re.IGNORECASE | re.DOTALL
                )
                title_match = title_pattern.search(text[:2000])
                
                if title_match:
                    title = f"{doc_type} {title_match.group(1).strip()}"
                    title = re.sub(r'\s+', ' ', title)
                else:
                    # Fallback: Use the first non-empty line as title
                    first_lines = [line.strip() for line in text.split("\n") if line.strip()]
                    if first_lines:
                        title = first_lines[0]
                
                break
        
        return {"document_type": doc_type, "title": title}
    
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
            r'\b(?:the\s+)?([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\s+\((?:the\s+)?"(?:Company|Seller|Buyer|Vendor|Client|Contractor|Consultant|Employer|Employee)"\)',
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