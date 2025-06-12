from academic_research_workflow import create_oates_research_system, AcademicResearchWorkflow
from pattern_matcher import create_default_pattern_matcher
from validation_system import create_cognitive_science_validation_system
from research_config import (
    create_cognitive_science_config,
    create_recursive_cognition_config,
    create_ethics_guidelines,
    create_journal_requirements,
    create_workflow_safeguards,
    get_default_config,
    validate_configuration
)
from collections import Dict, List
import time

fn demonstrate_research_identification():
    """Demonstrates identifying research contributions"""
    print("\n=== Research Identification Demo ===")
    
    # Create pattern matcher
    var matcher = create_default_pattern_matcher()
    
    # Sample research text
    var sample_text = """
    Mind-Wandering as Recursive Attention: A Fractal Analysis
    Oates, R., Smith, J., and Johnson, K.
    
    Abstract: This study investigates mind-wandering through the lens of recursive 
    cognition and fractal dynamics. Using fMRI analysis of default mode network 
    activity, we discovered scale-invariant patterns in attention fluctuations 
    during spontaneous thought. Our findings suggest that attention operates as 
    a self-similar system with recursive properties, providing new insights into 
    the attention-recognition decoupling phenomenon. These results have implications 
    for understanding meta-awareness and consciousness.
    
    Keywords: mind-wandering, recursive cognition, fractal analysis, DMN, attention dynamics
    """
    
    # Get Oates signature
    var oates_signature = matcher.signatures["oates_r"]
    
    # Find pattern matches
    var matches = matcher.match_patterns(sample_text, oates_signature)
    print("Found " + str(len(matches)) + " pattern matches")
    
    # Calculate confidence
    var confidence = matcher.calculate_aggregate_confidence(matches, oates_signature)
    print("Authorship confidence: " + str(confidence))
    
    # Extract metadata
    var metadata = matcher.extract_research_metadata(sample_text, matches)
    print("Extracted title: " + metadata.get("title", "Unknown"))
    print("Research domain: " + metadata.get("domain", "Unknown"))
    
    # Verify author contribution
    var is_author = matcher.identify_author_contribution(sample_text, "Oates, R.")
    print("Author contribution verified: " + str(is_author))

fn demonstrate_manuscript_development():
    """Demonstrates manuscript development workflow"""
    print("\n=== Manuscript Development Demo ===")
    
    # Create workflow system
    var workflow = create_oates_research_system()
    
    # Prepare research data
    var research_data = Dict[String, String]()
    research_data["title"] = "Recursive Meta-Awareness in Mind-Wandering: A Multiscale Analysis"
    research_data["background"] = "Previous research has shown that mind-wandering involves complex dynamics of attention and awareness"
    research_data["methods"] = "We employed fMRI scanning during rest and task conditions, analyzing data using fractal dimension calculations and recurrence quantification analysis"
    research_data["results"] = "We found evidence for scale-invariant patterns in DMN activity correlating with subjective reports of meta-awareness during mind-wandering episodes"
    research_data["conclusions"] = "These findings support a recursive model of consciousness where meta-awareness emerges from fractal dynamics in neural activity"
    research_data["framework"] = "recursive cognition"
    research_data["methodology"] = "neuroimaging, nonlinear dynamics analysis"
    research_data["keywords"] = "mind-wandering,meta-awareness,fractals,DMN,recursive cognition"
    research_data["co_authors"] = "Chen, S., Rodriguez, E."
    
    # Generate manuscript outline
    var manuscript = workflow.generate_manuscript_outline(research_data, "Consciousness and Cognition")
    
    print("Generated manuscript:")
    print("  Title: " + manuscript.title)
    print("  Authors: " + ", ".join(manuscript.authors))
    print("  Status: " + manuscript.publication_status)
    print("  Framework: " + manuscript.theoretical_framework)
    
    # Add to workflow
    workflow.research_papers.append(manuscript)

fn demonstrate_validation_process():
    """Demonstrates validation and review process"""
    print("\n=== Validation Process Demo ===")
    
    # Create validation system
    var validation_system = create_cognitive_science_validation_system()
    
    # Sample paper content for validation
    var paper_content = """
    Title: Fractal Dynamics in Mind-Wandering Episodes
    
    Methods: We recruited 45 participants (23 female, mean age 24.3) for this fMRI study.
    Power analysis indicated a minimum sample size of 40 for detecting medium effect sizes.
    Statistical analyses included repeated-measures ANOVA with Bonferroni correction for 
    multiple comparisons. Effect sizes (Cohen's d) and 95% confidence intervals are reported.
    
    Data Availability: All data and analysis code are available at https://osf.io/example
    Analysis scripts are documented in our GitHub repository with requirements.txt for 
    Python 3.8 environment specification.
    
    Results: We found significant differences in fractal dimension between mind-wandering 
    and focused attention states (F(1,44) = 23.4, p < 0.001, d = 0.82, 95% CI [0.45, 1.19]).
    """
    
    var paper_id = "paper_001"
    
    # Perform statistical validation
    var stat_validation = validation_system.validate_statistics(paper_id, paper_content)
    print("Statistical Validation:")
    print("  Sample size adequate: " + str(stat_validation.sample_size_adequate))
    print("  Power analysis provided: " + str(stat_validation.power_analysis_provided))
    print("  Effect sizes reported: " + str(stat_validation.effect_sizes_reported))
    print("  Multiple comparisons corrected: " + str(stat_validation.multiple_comparisons_corrected))
    print("  Overall score: " + str(stat_validation.validation_score))
    
    # Check reproducibility
    var repro_check = validation_system.check_reproducibility(paper_id, paper_content)
    print("\nReproducibility Check:")
    print("  Data available: " + str(repro_check.data_available))
    print("  Code available: " + str(repro_check.code_available))
    print("  Environment specified: " + str(repro_check.environment_specified))
    print("  Overall score: " + str(repro_check.reproducibility_score))
    
    # Assess quality
    var quality = validation_system.assess_quality(paper_id, paper_content)
    print("\nQuality Assessment:")
    print("  Clarity score: " + str(quality.clarity_score))
    print("  Methodology score: " + str(quality.methodology_score))
    print("  Overall quality: " + str(quality.overall_quality))

fn demonstrate_ethics_and_approval():
    """Demonstrates ethics compliance and approval workflow"""
    print("\n=== Ethics & Approval Demo ===")
    
    # Create workflow and ethics guidelines
    var workflow = create_oates_research_system()
    var ethics_guidelines = create_ethics_guidelines()
    
    # Create a paper entry
    var paper_id = "paper_dmn_001"
    
    # Create ethics compliance record
    var ethics = workflow.create_ethics_compliance_record(paper_id)
    print("Ethics compliance record created for paper: " + paper_id)
    
    # Show required ethics components
    print("\nRequired Ethics Components:")
    for guideline_name, guideline in ethics_guidelines.items():
        print("  " + guideline.guideline_name + ":")
        print("    Mandatory: " + str(guideline.mandatory))
        print("    Review process: " + guideline.review_process)
    
    # Initiate approval workflow
    var approvers = List[String](
        "oates_r",  # Primary author
        "chen_s",   # Co-author 1
        "rodriguez_e",  # Co-author 2
        "dept_chair_smith",  # Department chair
        "irb_committee"  # Ethics board
    )
    
    var approval_workflow = workflow.initiate_approval_workflow(paper_id, approvers, 14)
    print("\nApproval workflow initiated:")
    print("  Workflow ID: " + approval_workflow.workflow_id)
    print("  Required approvers: " + str(len(approvers)))
    print("  Timeout: " + str(approval_workflow.timeout_days) + " days")
    
    # Simulate some approvals (in practice, these would come from UI/API)
    print("\nSimulating approval decisions...")
    workflow.record_approval_decision(approval_workflow.workflow_id, "oates_r", "approved")
    workflow.record_approval_decision(approval_workflow.workflow_id, "chen_s", "approved")
    workflow.record_approval_decision(approval_workflow.workflow_id, "rodriguez_e", "approved")
    
    # Check publication readiness
    var readiness = workflow.check_publication_readiness(paper_id)
    print("\nPublication Readiness:")
    for check, status in readiness.items():
        print("  " + check + ": " + str(status))

fn demonstrate_safety_measures():
    """Demonstrates safety measures and human oversight"""
    print("\n=== Safety Measures Demo ===")
    
    # Get workflow configuration
    var config = get_default_config()
    print("Default Workflow Configuration:")
    print("  Minimum reviewers: " + str(config.min_reviewers))
    print("  Require unanimous approval: " + str(config.require_unanimous_approval))
    print("  Approval timeout: " + str(config.approval_timeout_days) + " days")
    print("  Revision cycles allowed: " + str(config.allow_revision_cycles))
    print("  Audit trail enabled: " + str(config.audit_trail_enabled))
    
    # Validate configuration
    var is_valid = validate_configuration(config)
    print("\nConfiguration validation: " + ("PASSED" if is_valid else "FAILED"))
    
    # Show safeguards
    var safeguards = create_workflow_safeguards()
    print("\nWorkflow Safeguards:")
    for safeguard, setting in safeguards.items():
        print("  " + safeguard + ": " + setting)
    
    print("\n⚠️  CRITICAL SAFEGUARDS:")
    print("  • No autonomous publication capability")
    print("  • All decisions require human approval")
    print("  • Comprehensive audit trail maintained")
    print("  • Timeout protection with automatic holds")
    print("  • Institutional compliance required")

fn demonstrate_journal_requirements():
    """Demonstrates journal-specific requirements"""
    print("\n=== Journal Requirements Demo ===")
    
    var journals = create_journal_requirements()
    
    for journal_key, requirements in journals.items():
        print("\n" + requirements.journal_name + ":")
        print("  Submission format: " + requirements.submission_format)
        print("  Peer review type: " + requirements.peer_review_type)
        print("  Open access: " + str(requirements.open_access))
        print("  Data sharing policy: " + requirements.data_sharing_policy)
        print("  Ethics requirements:")
        for req in requirements.ethics_requirements:
            print("    - " + req)

fn main():
    """Main demonstration of the academic research workflow system"""
    print("=====================================================")
    print("Academic Research Workflow System Demonstration")
    print("=====================================================")
    print("Author Focus: Oates, R. - Cognitive Science Research")
    print("Topics: Mind-wandering, Recursive Cognition, Attention Dynamics")
    print("")
    print("IMPORTANT: This system requires human oversight for ALL publications")
    print("=====================================================")
    
    # Run demonstrations
    demonstrate_research_identification()
    demonstrate_manuscript_development()
    demonstrate_validation_process()
    demonstrate_ethics_and_approval()
    demonstrate_safety_measures()
    demonstrate_journal_requirements()
    
    print("\n=====================================================")
    print("Demonstration Complete")
    print("=====================================================")
    print("\nKey Takeaways:")
    print("1. Pattern-based research identification with confidence scoring")
    print("2. Structured manuscript development with co-author management")
    print("3. Comprehensive validation including statistics and reproducibility")
    print("4. Multi-stakeholder approval workflows with timeout protection")
    print("5. Strict ethical compliance and human oversight requirements")
    print("6. Journal-specific formatting and requirements support")
    print("\n✓ This system assists researchers while maintaining academic integrity")
    print("✓ No research can be published without explicit human approval")
    print("✓ All ethical guidelines and institutional policies must be followed")