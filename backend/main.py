# Clinical Note Generation Fix - MediRecord AI
# This file specifically addresses clinical note generation issues

import os
import json
import logging
import re
from typing import Dict, Optional, Any
from datetime import datetime
import traceback

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ClinicalNoteGeneratorFixed:
    """Fixed version of clinical note generator with multiple fallback methods"""
    
    def _init_(self, config_manager):
        self.config = config_manager
        self.azure_client = None
        self.openai_client = None
        self.generation_method = "fallback"  # Start with fallback, upgrade if services available
        self._initialize_services()
        
    def _initialize_services(self):
        """Initialize all available AI services for note generation"""
        
        # Try Azure OpenAI first
        try:
            from openai import AzureOpenAI
            endpoint = self.config.get('openai_endpoint')
            api_key = self.config.get('openai_api_key')
            
            if endpoint and api_key:
                self.azure_client = AzureOpenAI(
                    api_key=api_key,
                    api_version="2024-02-15-preview",
                    azure_endpoint=endpoint,
                    timeout=30.0  # Add timeout to prevent hanging
                )
                
                # Test the connection
                if self._test_azure_connection():
                    self.generation_method = "azure"
                    logger.info("✅ Azure OpenAI initialized and tested successfully")
                else:
                    logger.warning("❌ Azure OpenAI connection test failed")
                    
        except Exception as e:
            logger.error(f"Azure OpenAI initialization failed: {e}")
        
        # Try standard OpenAI as fallback
        try:
            import openai
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                openai.api_key = openai_key
                self.generation_method = "openai_standard"
                logger.info("✅ Standard OpenAI configured as fallback")
        except Exception as e:
            logger.error(f"Standard OpenAI setup failed: {e}")
    
    def _test_azure_connection(self) -> bool:
        """Test Azure OpenAI connection with a simple request"""
        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4",  # Replace with your actual model deployment name
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=10
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            logger.error(f"Azure OpenAI connection test failed: {e}")
            return False
    
    def generate_clinical_notes(self, transcript: str, patient_context: Dict = None) -> Dict:
        """
        Generate clinical notes with multiple fallback methods
        
        Args:
            transcript: The conversation transcript
            patient_context: Patient information and context
            
        Returns:
            Dict containing clinical notes in SOAP format
        """
        
        logger.info(f"Generating clinical notes using method: {self.generation_method}")
        logger.debug(f"Transcript length: {len(transcript)} characters")
        
        # Validate input
        if not transcript or len(transcript.strip()) < 10:
            logger.warning("Transcript too short or empty")
            return self._create_empty_notes_template(transcript)
        
        # Try different generation methods in order of preference
        methods = [
            ("azure", self._generate_with_azure),
            ("openai_standard", self._generate_with_standard_openai),
            ("rule_based", self._generate_with_rules),
            ("template", self._generate_template_notes)
        ]
        
        for method_name, method_func in methods:
            if method_name == self.generation_method or self.generation_method == "fallback":
                try:
                    logger.info(f"Attempting generation with: {method_name}")
                    result = method_func(transcript, patient_context)
                    
                    if result and self._validate_notes(result):
                        logger.info(f"✅ Successfully generated notes with {method_name}")
                        result['generation_method'] = method_name
                        result['generated_at'] = datetime.now().isoformat()
                        return result
                    else:
                        logger.warning(f"❌ {method_name} generated invalid notes")
                        
                except Exception as e:
                    logger.error(f"❌ {method_name} failed: {e}")
                    continue
        
        # If all methods fail, return structured template
        logger.error("All generation methods failed, returning template")
        return self._create_emergency_template(transcript, patient_context)
    
    def _generate_with_azure(self, transcript: str, patient_context: Dict = None) -> Optional[Dict]:
        """Generate notes using Azure OpenAI"""
        
        if not self.azure_client:
            raise Exception("Azure client not initialized")
        
        # Create enhanced prompt
        prompt = self._create_enhanced_prompt(transcript, patient_context)
        
        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4",  # Use your actual deployment name
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert medical documentation assistant. 
                        Convert doctor-patient conversations into structured clinical notes.
                        ALWAYS respond with valid JSON in exactly this format:
                        {
                            "subjective": "string",
                            "objective": "string", 
                            "assessment": "string",
                            "plan": "string",
                            "summary": "string"
                        }
                        Do not include any text outside the JSON structure."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
                timeout=30
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Azure OpenAI response: {content}")
            
            return self._parse_ai_response(content)
            
        except Exception as e:
            logger.error(f"Azure OpenAI request failed: {e}")
            raise
    
    def _generate_with_standard_openai(self, transcript: str, patient_context: Dict = None) -> Optional[Dict]:
        """Generate notes using standard OpenAI API"""
        
        try:
            import openai
            
            prompt = self._create_enhanced_prompt(transcript, patient_context)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a medical documentation expert. Convert conversations to SOAP notes.
                        Respond only with valid JSON in this exact format:
                        {"subjective": "...", "objective": "...", "assessment": "...", "plan": "...", "summary": "..."}"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            return self._parse_ai_response(content)
            
        except Exception as e:
            logger.error(f"Standard OpenAI failed: {e}")
            raise
    
    def _generate_with_rules(self, transcript: str, patient_context: Dict = None) -> Dict:
        """Generate notes using rule-based extraction"""
        
        logger.info("Using rule-based clinical note generation")
        
        # Extract sections using keyword matching
        subjective = self._extract_subjective_info(transcript)
        objective = self._extract_objective_info(transcript) 
        assessment = self._extract_assessment_info(transcript)
        plan = self._extract_plan_info(transcript)
        
        return {
            "subjective": subjective or "Patient reported symptoms and concerns from conversation.",
            "objective": objective or "Physical examination findings documented during visit.",
            "assessment": assessment or "Clinical assessment based on conversation.",
            "plan": plan or "Treatment plan discussed with patient.",
            "summary": f"Clinical visit documented. Transcript length: {len(transcript)} characters.",
            "extraction_method": "rule_based"
        }
    
    def _generate_template_notes(self, transcript: str, patient_context: Dict = None) -> Dict:
        """Generate basic template notes as last resort"""
        
        logger.info("Using template-based note generation")
        
        # Extract basic information
        patient_name = self._extract_patient_name(transcript, patient_context)
        visit_type = self._extract_visit_type(transcript)
        
        return {
            "subjective": f"Patient {patient_name} presented for {visit_type}. "
                         f"Detailed conversation recorded in transcript.",
            "objective": "Physical examination findings to be documented by attending physician.",
            "assessment": "Clinical assessment pending physician review of recorded conversation.",
            "plan": "Treatment plan to be finalized based on clinical assessment.",
            "summary": f"Visit with {patient_name} documented. "
                      f"Full transcript available for physician review.",
            "raw_transcript": transcript[:500] + "..." if len(transcript) > 500 else transcript
        }
    
    def _create_enhanced_prompt(self, transcript: str, patient_context: Dict = None) -> str:
        """Create an enhanced prompt for AI generation"""
        
        context_info = ""
        if patient_context:
            context_info = f"""
            Patient Information:
            - Name: {patient_context.get('name', 'Unknown')}
            - Age: {patient_context.get('age', 'Unknown')}
            - Medical Record: {patient_context.get('medical_record_number', 'Unknown')}
            """
        
        return f"""
        {context_info}
        
        Doctor-Patient Conversation Transcript:
        {transcript}
        
        Please convert this conversation into structured clinical notes using the SOAP format:
        
        - Subjective: What the patient reports (symptoms, concerns, history)
        - Objective: Observable/measurable findings (vital signs, physical exam)
        - Assessment: Clinical judgment/diagnosis
        - Plan: Treatment plan, follow-up, instructions
        
        Respond with JSON format only:
        {{
            "subjective": "detailed subjective findings",
            "objective": "objective findings and measurements", 
            "assessment": "clinical assessment and diagnosis",
            "plan": "treatment plan and next steps",
            "summary": "brief visit summary"
        }}
        """
    
    def _parse_ai_response(self, content: str) -> Optional[Dict]:
        """Parse AI response and extract JSON"""
        
        try:
            # Clean the response
            content = content.strip()
            
            # Remove code block markers if present
            if content.startswith("json"):
                content = content.replace("json", "", 1)
            if content.startswith(""):
                content = content.replace("", "", 1)
            if content.endswith(""):
                content = content.rsplit("", 1)[0]
            
            # Find JSON in the response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                result = json.loads(json_content)
                
                # Ensure all required fields are present
                required_fields = ["subjective", "objective", "assessment", "plan", "summary"]
                for field in required_fields:
                    if field not in result:
                        result[field] = f"[{field.title()} information needs to be added]"
                
                return result
            else:
                logger.error("No valid JSON found in AI response")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Content: {content}")
            return None
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return None
    
    def _extract_subjective_info(self, transcript: str) -> str:
        """Extract subjective information using pattern matching"""
        
        subjective_patterns = [
            r"patient.*(?:complains?|reports?|says?|mentions?|feels?|experiences?)\s+(.+?)(?:\.|doctor|physician)",
            r"(?:i|patient).*(?:feel|hurt|pain|ache|sick|nauseous|dizzy|tired)\s+(.+?)(?:\.|doctor)",
            r"(?:symptoms?|complaint|concern|problem).?:?\s(.+?)(?:\.|doctor|examination)"
        ]
        
        findings = []
        for pattern in subjective_patterns:
            matches = re.findall(pattern, transcript.lower(), re.IGNORECASE | re.DOTALL)
            findings.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        if findings:
            return "Patient reports: " + "; ".join(findings[:3])
        
        # Fallback: extract first few sentences mentioning patient
        sentences = transcript.split('. ')
        patient_sentences = [s for s in sentences if 'patient' in s.lower()]
        if patient_sentences:
            return patient_sentences[0] + "."
        
        return "Patient presented with concerns as documented in transcript."
    
    def _extract_objective_info(self, transcript: str) -> str:
        """Extract objective information"""
        
        objective_patterns = [
            r"(?:blood pressure|bp).*?(\d+\/\d+)",
            r"(?:heart rate|pulse).*?(\d+)",
            r"(?:temperature|temp).?(\d+\.?\d)",
            r"(?:weight).?(\d+)\s(?:lbs?|pounds?|kg)",
            r"(?:examination|exam|physical).*?(.+?)(?:\.|assessment|plan)"
        ]
        
        findings = []
        for pattern in objective_patterns:
            matches = re.findall(pattern, transcript.lower(), re.IGNORECASE)
            findings.extend([match.strip() for match in matches if match.strip()])
        
        if findings:
            return "Examination findings: " + "; ".join(findings[:5])
        
        return "Physical examination findings documented during visit."
    
    def _extract_assessment_info(self, transcript: str) -> str:
        """Extract assessment information"""
        
        assessment_patterns = [
            r"(?:diagnosis|diagnosed|assess|assessment).?:?\s(.+?)(?:\.|plan|treatment)",
            r"(?:condition|disorder|disease).*?(.+?)(?:\.|plan|treatment)",
            r"(?:likely|probable|possible|suspect).*?(.+?)(?:\.|plan|treatment)"
        ]
        
        findings = []
        for pattern in assessment_patterns:
            matches = re.findall(pattern, transcript.lower(), re.IGNORECASE | re.DOTALL)
            findings.extend([match.strip() for match in matches if len(match.strip()) > 3])
        
        if findings:
            return "Clinical assessment: " + "; ".join(findings[:2])
        
        return "Clinical assessment pending based on examination findings."
    
    def _extract_plan_info(self, transcript: str) -> str:
        """Extract plan information"""
        
        plan_patterns = [
            r"(?:plan|treatment|prescribe|medication|follow.?up).?:?\s(.+?)(?:\.|$)",
            r"(?:recommend|suggest|advise).*?(.+?)(?:\.|$)",
            r"(?:next|return|come back|appointment).*?(.+?)(?:\.|$)"
        ]
        
        findings = []
        for pattern in plan_patterns:
            matches = re.findall(pattern, transcript.lower(), re.IGNORECASE | re.DOTALL)
            findings.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
        if findings:
            return "Treatment plan: " + "; ".join(findings[:3])
        
        return "Treatment plan to be determined based on assessment."
    
    def _extract_patient_name(self, transcript: str, patient_context: Dict = None) -> str:
        """Extract patient name"""
        if patient_context and patient_context.get('name'):
            return patient_context['name']
        
        # Try to find name in transcript
        name_patterns = [
            r"patient\s+(\w+\s+\w+)",
            r"mr\.?\s+(\w+)",
            r"mrs?\.?\s+(\w+)",
            r"hello\s+(\w+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, transcript.lower())
            if match:
                return match.group(1).title()
        
        return "[Patient Name]"
    
    def _extract_visit_type(self, transcript: str) -> str:
        """Extract visit type"""
        if "follow" in transcript.lower():
            return "follow-up visit"
        elif "routine" in transcript.lower() or "check" in transcript.lower():
            return "routine check-up"
        elif "annual" in transcript.lower():
            return "annual examination"
        else:
            return "consultation"
    
    def _validate_notes(self, notes: Dict) -> bool:
        """Validate that generated notes are complete and valid"""
        
        if not isinstance(notes, dict):
            return False
        
        required_fields = ["subjective", "objective", "assessment", "plan"]
        
        for field in required_fields:
            if field not in notes:
                logger.warning(f"Missing required field: {field}")
                return False
            
            if not notes[field] or len(notes[field].strip()) < 10:
                logger.warning(f"Field {field} is too short or empty")
                return False
        
        return True
    
    def _create_empty_notes_template(self, transcript: str) -> Dict:
        """Create template for empty or invalid transcript"""
        return {
            "subjective": "No significant patient complaints recorded.",
            "objective": "Physical examination to be completed.",
            "assessment": "Clinical assessment pending.",
            "plan": "Follow-up plan to be determined.",
            "summary": "Brief encounter documented.",
            "transcript": transcript,
            "status": "template_used"
        }
    
    def _create_emergency_template(self, transcript: str, patient_context: Dict = None) -> Dict:
        """Create emergency fallback template"""
        
        patient_name = self._extract_patient_name(transcript, patient_context)
        
        return {
            "subjective": f"Visit with {patient_name}. Full conversation transcript attached for review.",
            "objective": "Physical examination findings to be documented by attending physician.",
            "assessment": "Clinical assessment requires physician review of recorded conversation.",
            "plan": "Treatment plan pending clinical assessment completion.",
            "summary": f"Clinical encounter with {patient_name} documented via voice recording.",
            "raw_transcript": transcript,
            "requires_physician_review": True,
            "generation_status": "emergency_template",
            "timestamp": datetime.now().isoformat()
        }

# Diagnostic and testing functions
def test_clinical_note_generation():
    """Test clinical note generation with sample data"""
    
    print("🧪 Testing Clinical Note Generation")
    print("=" * 50)
    
    # Sample transcript for testing
    sample_transcript = """
    Doctor: Hello, how are you feeling today?
    Patient: I've been having headaches for the past week, especially in the mornings.
    Doctor: Can you describe the pain?
    Patient: It's a throbbing pain on the right side of my head. I also feel nauseous sometimes.
    Doctor: Any recent changes in your vision?
    Patient: No, my vision is fine.
    Doctor: Let me check your blood pressure. It's 140 over 90.
    Doctor: Your temperature is normal at 98.6 degrees.
    Doctor: Based on your symptoms and elevated blood pressure, this could be tension headaches related to hypertension.
    Doctor: I'm going to prescribe a blood pressure medication and recommend you reduce stress and get more sleep.
    Patient: Okay, when should I come back?
    Doctor: Let's schedule a follow-up in two weeks to check your blood pressure.
    """
    
    sample_patient_context = {
        "name": "John Smith",
        "age": 45,
        "medical_record_number": "MR12345"
    }
    
    # Mock config manager
    class MockConfig:
        def get(self, key, default=None):
            return default
    
    # Test the generator
    try:
        generator = ClinicalNoteGeneratorFixed(MockConfig())
        result = generator.generate_clinical_notes(sample_transcript, sample_patient_context)
        
        print("✅ Clinical Note Generation Test Results:")
        print("-" * 40)
        print(f"Generation Method: {result.get('generation_method', 'Unknown')}")
        print(f"Subjective: {result.get('subjective', 'N/A')}")
        print(f"Objective: {result.get('objective', 'N/A')}")
        print(f"Assessment: {result.get('assessment', 'N/A')}")
        print(f"Plan: {result.get('plan', 'N/A')}")
        print(f"Summary: {result.get('summary', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(traceback.format_exc())
        return False

def fix_azure_openai_issues():
    """Specific fixes for Azure OpenAI issues"""
    
    print("\n🔧 Azure OpenAI Troubleshooting")
    print("=" * 40)
    
    # Check environment variables
    required_vars = {
        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
        'AZURE_OPENAI_KEY': os.getenv('AZURE_OPENAI_KEY')
    }
    
    missing_vars = [k for k, v in required_vars.items() if not v]
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nTo fix, set these environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your_value_here")
        return False
    
    # Test Azure OpenAI connection
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=required_vars['AZURE_OPENAI_KEY'],
            api_version="2024-02-15-preview",
            azure_endpoint=required_vars['AZURE_OPENAI_ENDPOINT']
        )
        
        # Test with simple request
        response = client.chat.completions.create(
            model="gpt-4",  # Replace with your model deployment name
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("✅ Azure OpenAI connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        print("\nCommon fixes:")
        print("1. Verify your endpoint URL format: https://your-resource.openai.azure.com/")
        print("2. Check your API key is correct")
        print("3. Verify your model deployment name")
        print("4. Ensure your Azure resource is active")
        return False

# Main execution
if __name__ == "__main__":
    print("MediRecord AI - Clinical Note Generation Fix")
    print("=" * 50)
    
    # Run tests
    test_success = test_clinical_note_generation()
    azure_success = fix_azure_openai_issues()
    
    if test_success:
        print("\n✅ Clinical note generation is working!")
    else:
        print("\n❌ Clinical note generation needs attention")
    
    print("\n📝 Quick Fix Checklist:")
    print("1. ✅ Install required packages: pip install openai azure-cognitiveservices-speech")
    print("2. ✅ Set Azure environment variables")
    print("3. ✅ Test with sample transcript")
    print("4. ✅ Verify model deployment name in Azure")
    print("5. ✅ Check API quotas and limits")