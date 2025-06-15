from collections import Dict, List

struct ResearchDomainConfig:
    var domain_name: String
    var key_concepts: List[String]
    var methodologies: List[String]
    var journals: List[String]
    var ethical_considerations: List[String]
    
    fn __init__(inout self, name: String):
        self.domain_name = name
        self.key_concepts = List[String]()
        self.methodologies = List[String]()
        self.journals = List[String]()
        self.ethical_considerations = List[String]()

struct EthicalGuidelinesConfig:
    var guideline_name: String
    var requirements: List[String]
    var mandatory: Bool
    var review_process: String
    var documentation_needed: List[String]
    
    fn __init__(inout self, name: String, mandatory: Bool = True):
        self.guideline_name = name
        self.requirements = List[String]()
        self.mandatory = mandatory
        self.review_process = "standard"
        self.documentation_needed = List[String]()

struct JournalRequirements:
    var journal_name: String
    var submission_format: String
    var peer_review_type: String
    var ethics_requirements: List[String]
    var open_access: Bool
    var data_sharing_policy: String
    
    fn __init__(inout self, name: String):
        self.journal_name = name
        self.submission_format = "standard"
        self.peer_review_type = "double-blind"
        self.ethics_requirements = List[String]()
        self.open_access = False
        self.data_sharing_policy = "encouraged"

struct ResearchWorkflowConfig:
    var min_reviewers: Int
    var approval_timeout_days: Int
    var require_unanimous_approval: Bool
    var allow_revision_cycles: Int
    var data_retention_days: Int
    var audit_trail_enabled: Bool
    
    fn __init__(inout self):
        self.min_reviewers = 3
        self.approval_timeout_days = 14
        self.require_unanimous_approval = True
        self.allow_revision_cycles = 3
        self.data_retention_days = 365
        self.audit_trail_enabled = True

fn create_cognitive_science_config() -> ResearchDomainConfig:
    """Creates configuration for cognitive science research domain"""
    var config = ResearchDomainConfig("Cognitive Science")
    
    config.key_concepts = List[String](
        "mind-wandering", "attention", "consciousness", "metacognition",
        "default mode network", "executive control", "working memory",
        "cognitive flexibility", "task-switching", "attention regulation"
    )
    
    config.methodologies = List[String](
        "fMRI neuroimaging", "EEG recording", "behavioral experiments",
        "computational modeling", "experience sampling", "meta-analysis",
        "psychophysics", "eye tracking", "reaction time analysis"
    )
    
    config.journals = List[String](
        "Cognitive Science", "Journal of Experimental Psychology",
        "Consciousness and Cognition", "NeuroImage", "Brain and Cognition",
        "Psychological Science", "Current Opinion in Behavioral Sciences"
    )
    
    config.ethical_considerations = List[String](
        "informed consent required", "IRB approval mandatory",
        "participant anonymity", "data protection compliance",
        "vulnerable population protocols", "deception disclosure"
    )
    
    return config

fn create_recursive_cognition_config() -> ResearchDomainConfig:
    """Creates configuration for recursive cognition research"""
    var config = ResearchDomainConfig("Recursive Cognition")
    
    config.key_concepts = List[String](
        "recursive processing", "meta-awareness", "self-reference",
        "hierarchical cognition", "fractal dynamics", "scale invariance",
        "nested attention", "recursive loops", "meta-cognitive monitoring"
    )
    
    config.methodologies = List[String](
        "fractal analysis", "nonlinear dynamics modeling",
        "hierarchical Bayesian modeling", "recurrence quantification",
        "multi-scale entropy analysis", "phase space reconstruction"
    )
    
    config.journals = List[String](
        "Cognitive Systems Research", "Frontiers in Psychology",
        "Nonlinear Dynamics Psychology and Life Sciences",
        "Philosophical Psychology", "Topics in Cognitive Science"
    )
    
    config.ethical_considerations = List[String](
        "complexity disclosure to participants",
        "computational resource sharing",
        "algorithm transparency requirements"
    )
    
    return config

fn create_ethics_guidelines() -> Dict[String, EthicalGuidelinesConfig]:
    """Creates comprehensive ethical guidelines configuration"""
    var guidelines = Dict[String, EthicalGuidelinesConfig]()
    
    # IRB Approval
    var irb = EthicalGuidelinesConfig("IRB Approval", mandatory=True)
    irb.requirements = List[String](
        "full protocol submission", "risk assessment completed",
        "consent forms approved", "data management plan",
        "participant recruitment strategy", "debriefing procedures"
    )
    irb.review_process = "institutional"
    irb.documentation_needed = List[String](
        "research protocol", "consent forms", "recruitment materials",
        "data protection plan", "risk mitigation strategies"
    )
    guidelines["irb_approval"] = irb
    
    # Data Privacy
    var privacy = EthicalGuidelinesConfig("Data Privacy Compliance", mandatory=True)
    privacy.requirements = List[String](
        "GDPR compliance", "anonymization procedures",
        "secure storage protocols", "access control measures",
        "retention period specification", "deletion procedures"
    )
    privacy.review_process = "technical_and_legal"
    privacy.documentation_needed = List[String](
        "data protection impact assessment", "anonymization protocol",
        "security measures documentation", "access log requirements"
    )
    guidelines["data_privacy"] = privacy
    
    # Author Consent
    var consent = EthicalGuidelinesConfig("Author Consent", mandatory=True)
    consent.requirements = List[String](
        "all authors approve submission", "authorship order agreed",
        "contribution statements provided", "conflict disclosure",
        "copyright agreement signed", "open access decision"
    )
    consent.review_process = "collaborative"
    consent.documentation_needed = List[String](
        "author agreement form", "contribution statements",
        "conflict of interest forms", "copyright transfer"
    )
    guidelines["author_consent"] = consent
    
    # Institutional Approval
    var institutional = EthicalGuidelinesConfig("Institutional Approval", mandatory=True)
    institutional.requirements = List[String](
        "department chair approval", "research compliance check",
        "resource allocation confirmed", "liability coverage",
        "institutional affiliation accurate", "funding compliance"
    )
    institutional.review_process = "administrative"
    institutional.documentation_needed = List[String](
        "department approval form", "compliance checklist",
        "funding disclosure", "institutional letter"
    )
    guidelines["institutional_approval"] = institutional
    
    return guidelines

fn create_journal_requirements() -> Dict[String, JournalRequirements]:
    """Creates journal-specific requirements configuration"""
    var journals = Dict[String, JournalRequirements]()
    
    # Cognitive Science Journal
    var cogsci = JournalRequirements("Cognitive Science")
    cogsci.submission_format = "LaTeX preferred"
    cogsci.peer_review_type = "double-blind"
    cogsci.ethics_requirements = List[String](
        "IRB approval number", "ethics statement required",
        "data availability statement", "preregistration encouraged"
    )
    cogsci.open_access = True
    cogsci.data_sharing_policy = "mandatory"
    journals["cognitive_science"] = cogsci
    
    # NeuroImage
    var neuroimage = JournalRequirements("NeuroImage")
    neuroimage.submission_format = "Word or LaTeX"
    neuroimage.peer_review_type = "single-blind"
    neuroimage.ethics_requirements = List[String](
        "ethics committee approval", "participant consent confirmation",
        "clinical trial registration if applicable"
    )
    neuroimage.open_access = True
    neuroimage.data_sharing_policy = "mandatory with exceptions"
    journals["neuroimage"] = neuroimage
    
    # Consciousness and Cognition
    var consciousness = JournalRequirements("Consciousness and Cognition")
    consciousness.submission_format = "Word preferred"
    consciousness.peer_review_type = "double-blind"
    consciousness.ethics_requirements = List[String](
        "ethics approval required", "consent process description",
        "vulnerable population safeguards if applicable"
    )
    consciousness.open_access = False
    consciousness.data_sharing_policy = "encouraged"
    journals["consciousness_cognition"] = consciousness
    
    return journals

fn create_workflow_safeguards() -> Dict[String, String]:
    """Creates workflow safeguard configurations"""
    var safeguards = Dict[String, String]()
    
    safeguards["autonomous_publication"] = "strictly_prohibited"
    safeguards["human_oversight"] = "mandatory_all_stages"
    safeguards["approval_requirement"] = "unanimous_required"
    safeguards["revision_limit"] = "3_cycles_maximum"
    safeguards["timeout_action"] = "automatic_hold"
    safeguards["conflict_resolution"] = "escalate_to_committee"
    safeguards["audit_requirement"] = "comprehensive_trail"
    safeguards["data_retention"] = "secure_one_year_minimum"
    
    return safeguards

fn get_default_config() -> ResearchWorkflowConfig:
    """Returns default workflow configuration with safety measures"""
    return ResearchWorkflowConfig()

fn validate_configuration(config: ResearchWorkflowConfig) -> Bool:
    """Validates that configuration meets minimum safety requirements"""
    if config.min_reviewers < 3:
        print("ERROR: Minimum 3 reviewers required for safety")
        return False
    
    if not config.require_unanimous_approval:
        print("WARNING: Unanimous approval recommended for ethical compliance")
    
    if config.approval_timeout_days > 30:
        print("WARNING: Long timeout may delay important safety checks")
    
    if not config.audit_trail_enabled:
        print("ERROR: Audit trail required for compliance")
        return False
    
    return True