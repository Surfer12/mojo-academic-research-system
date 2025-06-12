from python import Python
from tensor import Tensor
from collections import Dict, List
from algorithm import vectorize, parallelize
import time
import math

struct PeerReviewer:
    var reviewer_id: String
    var name: String
    var expertise_areas: List[String]
    var institution: String
    var h_index: Int
    var review_count: Int
    var avg_review_time_days: Float32
    
    fn __init__(inout self, reviewer_id: String, name: String, 
                expertise: List[String], institution: String):
        self.reviewer_id = reviewer_id
        self.name = name
        self.expertise_areas = expertise
        self.institution = institution
        self.h_index = 0
        self.review_count = 0
        self.avg_review_time_days = 14.0

struct ReviewCriteria:
    var criterion_id: String
    var name: String
    var description: String
    var weight: Float32
    var required: Bool
    var evaluation_scale: Int
    
    fn __init__(inout self, criterion_id: String, name: String, 
                description: String, weight: Float32 = 1.0, 
                required: Bool = True, scale: Int = 5):
        self.criterion_id = criterion_id
        self.name = name
        self.description = description
        self.weight = weight
        self.required = required
        self.evaluation_scale = scale

struct ReviewEvaluation:
    var criterion_id: String
    var score: Int
    var comments: String
    var suggestions: List[String]
    var critical_issues: List[String]
    
    fn __init__(inout self, criterion_id: String):
        self.criterion_id = criterion_id
        self.score = 0
        self.comments = ""
        self.suggestions = List[String]()
        self.critical_issues = List[String]()

struct PeerReview:
    var review_id: String
    var paper_id: String
    var reviewer_id: String
    var review_date: Int
    var overall_recommendation: String
    var confidence_level: Float32
    var evaluations: List[ReviewEvaluation]
    var general_comments: String
    var revision_requirements: List[String]
    
    fn __init__(inout self, review_id: String, paper_id: String, reviewer_id: String):
        self.review_id = review_id
        self.paper_id = paper_id
        self.reviewer_id = reviewer_id
        self.review_date = int(time.time())
        self.overall_recommendation = "pending"
        self.confidence_level = 0.0
        self.evaluations = List[ReviewEvaluation]()
        self.general_comments = ""
        self.revision_requirements = List[String]()

struct StatisticalValidation:
    var paper_id: String
    var sample_size_adequate: Bool
    var power_analysis_provided: Bool
    var statistical_tests_appropriate: Bool
    var multiple_comparisons_corrected: Bool
    var effect_sizes_reported: Bool
    var confidence_intervals_provided: Bool
    var assumptions_tested: Bool
    var outliers_addressed: Bool
    var validation_score: Float32
    var issues: List[String]
    
    fn __init__(inout self, paper_id: String):
        self.paper_id = paper_id
        self.sample_size_adequate = False
        self.power_analysis_provided = False
        self.statistical_tests_appropriate = False
        self.multiple_comparisons_corrected = False
        self.effect_sizes_reported = False
        self.confidence_intervals_provided = False
        self.assumptions_tested = False
        self.outliers_addressed = False
        self.validation_score = 0.0
        self.issues = List[String]()

struct ReproducibilityCheck:
    var paper_id: String
    var data_available: Bool
    var data_format_documented: Bool
    var code_available: Bool
    var code_documented: Bool
    var environment_specified: Bool
    var analysis_steps_clear: Bool
    var results_reproducible: Bool
    var materials_available: Bool
    var protocol_detailed: Bool
    var reproducibility_score: Float32
    
    fn __init__(inout self, paper_id: String):
        self.paper_id = paper_id
        self.data_available = False
        self.data_format_documented = False
        self.code_available = False
        self.code_documented = False
        self.environment_specified = False
        self.analysis_steps_clear = False
        self.results_reproducible = False
        self.materials_available = False
        self.protocol_detailed = False
        self.reproducibility_score = 0.0

struct QualityMetrics:
    var clarity_score: Float32
    var originality_score: Float32
    var significance_score: Float32
    var methodology_score: Float32
    var presentation_score: Float32
    var references_score: Float32
    var overall_quality: Float32
    
    fn __init__(inout self):
        self.clarity_score = 0.0
        self.originality_score = 0.0
        self.significance_score = 0.0
        self.methodology_score = 0.0
        self.presentation_score = 0.0
        self.references_score = 0.0
        self.overall_quality = 0.0

struct ValidationSystem:
    var review_criteria: List[ReviewCriteria]
    var reviewer_pool: Dict[String, PeerReviewer]
    var active_reviews: Dict[String, List[PeerReview]]
    var statistical_validations: Dict[String, StatisticalValidation]
    var reproducibility_checks: Dict[String, ReproducibilityCheck]
    var quality_assessments: Dict[String, QualityMetrics]
    
    fn __init__(inout self):
        self.review_criteria = self._initialize_review_criteria()
        self.reviewer_pool = Dict[String, PeerReviewer]()
        self.active_reviews = Dict[String, List[PeerReview]]()
        self.statistical_validations = Dict[String, StatisticalValidation]()
        self.reproducibility_checks = Dict[String, ReproducibilityCheck]()
        self.quality_assessments = Dict[String, QualityMetrics]()
    
    fn _initialize_review_criteria(self) -> List[ReviewCriteria]:
        """Initializes standard review criteria for cognitive science papers"""
        var criteria = List[ReviewCriteria]()
        
        criteria.append(ReviewCriteria(
            "theoretical_contribution",
            "Theoretical Contribution",
            "Novel theoretical insights and framework development",
            weight=2.0,
            required=True
        ))
        
        criteria.append(ReviewCriteria(
            "methodological_rigor",
            "Methodological Rigor",
            "Appropriateness and quality of research methods",
            weight=2.5,
            required=True
        ))
        
        criteria.append(ReviewCriteria(
            "statistical_validity",
            "Statistical Validity",
            "Correctness and appropriateness of statistical analyses",
            weight=2.0,
            required=True
        ))
        
        criteria.append(ReviewCriteria(
            "clarity_presentation",
            "Clarity of Presentation",
            "Clear writing, logical flow, and effective communication",
            weight=1.5,
            required=True
        ))
        
        criteria.append(ReviewCriteria(
            "literature_review",
            "Literature Review",
            "Comprehensive and current review of relevant literature",
            weight=1.0,
            required=True
        ))
        
        criteria.append(ReviewCriteria(
            "ethical_compliance",
            "Ethical Compliance",
            "Adherence to ethical guidelines and proper consent procedures",
            weight=3.0,
            required=True
        ))
        
        criteria.append(ReviewCriteria(
            "data_transparency",
            "Data Transparency",
            "Availability and documentation of data and materials",
            weight=1.5,
            required=False
        ))
        
        return criteria
    
    fn assign_reviewers(inout self, paper_id: String, paper_content: String, 
                       num_reviewers: Int = 3) -> List[String]:
        """Assigns appropriate reviewers based on expertise matching"""
        var assigned_reviewers = List[String]()
        var expertise_scores = Dict[String, Float32]()
        
        # Extract paper topics for matching
        var paper_topics = self._extract_paper_topics(paper_content)
        
        # Score each potential reviewer
        for reviewer_id, reviewer in self.reviewer_pool.items():
            var score = self._calculate_expertise_match(paper_topics, reviewer.expertise_areas)
            expertise_scores[reviewer_id] = score
        
        # Sort reviewers by expertise match
        var sorted_reviewers = self._sort_by_scores(expertise_scores)
        
        # Assign top reviewers with conflict checking
        var assigned_count = 0
        for reviewer_id in sorted_reviewers:
            if assigned_count >= num_reviewers:
                break
            
            if self._check_no_conflicts(paper_id, reviewer_id):
                assigned_reviewers.append(reviewer_id)
                assigned_count += 1
        
        return assigned_reviewers
    
    fn _extract_paper_topics(self, content: String) -> List[String]:
        """Extracts main topics from paper content"""
        var topics = List[String]()
        
        var topic_keywords = [
            "mind-wandering", "attention", "consciousness", "DMN",
            "recursive", "fractal", "meta-awareness", "cognition",
            "neuroscience", "fMRI", "EEG", "computational"
        ]
        
        for keyword in topic_keywords:
            if keyword.lower() in content.lower():
                topics.append(keyword)
        
        return topics
    
    fn _calculate_expertise_match(self, paper_topics: List[String], 
                                expertise: List[String]) -> Float32:
        """Calculates expertise match score"""
        var match_count = 0
        
        for topic in paper_topics:
            for expert_area in expertise:
                if topic.lower() in expert_area.lower() or expert_area.lower() in topic.lower():
                    match_count += 1
        
        if len(paper_topics) > 0:
            return Float32(match_count) / Float32(len(paper_topics))
        return 0.0
    
    fn _sort_by_scores(self, scores: Dict[String, Float32]) -> List[String]:
        """Sorts reviewers by expertise scores"""
        var sorted_ids = List[String]()
        
        # Simple bubble sort for small reviewer pools
        var items = list(scores.items())
        var n = len(items)
        
        for i in range(n):
            for j in range(0, n-i-1):
                if items[j][1] < items[j+1][1]:
                    items[j], items[j+1] = items[j+1], items[j]
        
        for item in items:
            sorted_ids.append(item[0])
        
        return sorted_ids
    
    fn _check_no_conflicts(self, paper_id: String, reviewer_id: String) -> Bool:
        """Checks for conflicts of interest"""
        # In real implementation, would check:
        # - Same institution
        # - Recent collaboration
        # - Competing interests
        # For now, return True (no conflicts)
        return True
    
    fn create_peer_review(inout self, paper_id: String, reviewer_id: String) -> PeerReview:
        """Creates a new peer review instance"""
        var review_id = str(hash(paper_id + reviewer_id + str(time.time())))
        var review = PeerReview(review_id, paper_id, reviewer_id)
        
        # Initialize evaluations for each criterion
        for criterion in self.review_criteria:
            var evaluation = ReviewEvaluation(criterion.criterion_id)
            review.evaluations.append(evaluation)
        
        # Add to active reviews
        if paper_id not in self.active_reviews:
            self.active_reviews[paper_id] = List[PeerReview]()
        self.active_reviews[paper_id].append(review)
        
        return review
    
    fn validate_statistics(inout self, paper_id: String, paper_content: String) -> StatisticalValidation:
        """Performs statistical validation of paper"""
        var validation = StatisticalValidation(paper_id)
        
        # Check for statistical indicators in content
        var content_lower = paper_content.lower()
        
        # Sample size check
        if "n =" in content_lower or "sample size" in content_lower:
            validation.sample_size_adequate = self._check_sample_size(paper_content)
        
        # Power analysis
        validation.power_analysis_provided = "power analysis" in content_lower or "statistical power" in content_lower
        
        # Statistical tests
        var stat_tests = ["t-test", "anova", "regression", "correlation", "chi-square"]
        validation.statistical_tests_appropriate = any(test in content_lower for test in stat_tests)
        
        # Multiple comparisons
        validation.multiple_comparisons_corrected = any(
            correction in content_lower 
            for correction in ["bonferroni", "fdr", "false discovery rate", "multiple comparisons"]
        )
        
        # Effect sizes
        validation.effect_sizes_reported = any(
            effect in content_lower 
            for effect in ["effect size", "cohen's d", "eta squared", "r squared"]
        )
        
        # Confidence intervals
        validation.confidence_intervals_provided = "confidence interval" in content_lower or "ci" in content_lower
        
        # Assumptions
        validation.assumptions_tested = any(
            assumption in content_lower 
            for assumption in ["normality", "homogeneity", "sphericity", "assumptions"]
        )
        
        # Outliers
        validation.outliers_addressed = "outlier" in content_lower
        
        # Calculate overall score
        validation.validation_score = self._calculate_statistical_score(validation)
        
        # Identify issues
        if not validation.sample_size_adequate:
            validation.issues.append("Sample size may be inadequate")
        if not validation.power_analysis_provided:
            validation.issues.append("No power analysis provided")
        if not validation.effect_sizes_reported:
            validation.issues.append("Effect sizes not reported")
        
        self.statistical_validations[paper_id] = validation
        return validation
    
    fn _check_sample_size(self, content: String) -> Bool:
        """Checks if sample size is adequate"""
        # Simplified check - look for sample size mentions
        import re
        
        # Look for patterns like "n = 30" or "N=50"
        pattern = r'[nN]\s*=\s*(\d+)'
        matches = re.findall(pattern, content)
        
        if matches:
            for match in matches:
                if int(match) >= 20:  # Minimum threshold
                    return True
        
        return False
    
    fn _calculate_statistical_score(self, validation: StatisticalValidation) -> Float32:
        """Calculates overall statistical validation score"""
        var score: Float32 = 0.0
        var max_score: Float32 = 8.0
        
        if validation.sample_size_adequate:
            score += 1.0
        if validation.power_analysis_provided:
            score += 1.0
        if validation.statistical_tests_appropriate:
            score += 1.0
        if validation.multiple_comparisons_corrected:
            score += 1.0
        if validation.effect_sizes_reported:
            score += 1.0
        if validation.confidence_intervals_provided:
            score += 1.0
        if validation.assumptions_tested:
            score += 1.0
        if validation.outliers_addressed:
            score += 1.0
        
        return score / max_score
    
    fn check_reproducibility(inout self, paper_id: String, paper_content: String) -> ReproducibilityCheck:
        """Checks reproducibility aspects of the paper"""
        var check = ReproducibilityCheck(paper_id)
        var content_lower = paper_content.lower()
        
        # Data availability
        check.data_available = any(
            indicator in content_lower 
            for indicator in ["data availability", "open data", "dataset available", "osf.io", "github.com/data"]
        )
        
        # Data documentation
        check.data_format_documented = any(
            format_mention in content_lower 
            for format_mention in ["data format", "csv", "json", "hdf5", "data structure"]
        )
        
        # Code availability
        check.code_available = any(
            code_indicator in content_lower 
            for code_indicator in ["code available", "github.com", "gitlab", "source code", "analysis scripts"]
        )
        
        # Code documentation
        check.code_documented = "readme" in content_lower or "documentation" in content_lower
        
        # Environment specification
        check.environment_specified = any(
            env in content_lower 
            for env in ["python version", "r version", "package versions", "requirements.txt", "environment.yml"]
        )
        
        # Analysis clarity
        check.analysis_steps_clear = "analysis pipeline" in content_lower or "processing steps" in content_lower
        
        # Materials availability
        check.materials_available = "materials available" in content_lower or "stimulus materials" in content_lower
        
        # Protocol detail
        check.protocol_detailed = "detailed protocol" in content_lower or "step-by-step" in content_lower
        
        # Calculate reproducibility score
        check.reproducibility_score = self._calculate_reproducibility_score(check)
        
        self.reproducibility_checks[paper_id] = check
        return check
    
    fn _calculate_reproducibility_score(self, check: ReproducibilityCheck) -> Float32:
        """Calculates overall reproducibility score"""
        var score: Float32 = 0.0
        var max_score: Float32 = 9.0
        
        if check.data_available:
            score += 2.0  # High weight for data
        if check.data_format_documented:
            score += 1.0
        if check.code_available:
            score += 2.0  # High weight for code
        if check.code_documented:
            score += 1.0
        if check.environment_specified:
            score += 1.0
        if check.analysis_steps_clear:
            score += 1.0
        if check.materials_available:
            score += 0.5
        if check.protocol_detailed:
            score += 0.5
        
        return score / max_score
    
    fn assess_quality(inout self, paper_id: String, paper_content: String) -> QualityMetrics:
        """Assesses overall quality metrics of the paper"""
        var metrics = QualityMetrics()
        
        # Simplified quality assessment based on content analysis
        var content_lower = paper_content.lower()
        var word_count = len(paper_content.split())
        
        # Clarity score based on structure indicators
        var structure_elements = ["introduction", "methods", "results", "discussion", "abstract"]
        var found_elements = sum(1 for elem in structure_elements if elem in content_lower)
        metrics.clarity_score = Float32(found_elements) / Float32(len(structure_elements))
        
        # Originality score based on novel terms and concepts
        var originality_indicators = ["novel", "new approach", "first time", "innovative", "unique contribution"]
        var originality_count = sum(1 for indicator in originality_indicators if indicator in content_lower)
        metrics.originality_score = min(Float32(originality_count) / 3.0, 1.0)
        
        # Significance score based on impact language
        var significance_indicators = ["important", "significant", "major finding", "breakthrough", "implications"]
        var significance_count = sum(1 for indicator in significance_indicators if indicator in content_lower)
        metrics.significance_score = min(Float32(significance_count) / 3.0, 1.0)
        
        # Methodology score (already validated separately)
        if paper_id in self.statistical_validations:
            metrics.methodology_score = self.statistical_validations[paper_id].validation_score
        
        # Presentation score based on length and structure
        if 3000 <= word_count <= 10000:  # Reasonable paper length
            metrics.presentation_score = 0.8
        else:
            metrics.presentation_score = 0.5
        
        # References score based on citation indicators
        var reference_count = paper_content.lower().count("et al") + paper_content.count("(19") + paper_content.count("(20")
        if reference_count > 20:
            metrics.references_score = 0.9
        elif reference_count > 10:
            metrics.references_score = 0.7
        else:
            metrics.references_score = 0.4
        
        # Calculate overall quality
        metrics.overall_quality = (
            metrics.clarity_score * 0.15 +
            metrics.originality_score * 0.20 +
            metrics.significance_score * 0.20 +
            metrics.methodology_score * 0.25 +
            metrics.presentation_score * 0.10 +
            metrics.references_score * 0.10
        )
        
        self.quality_assessments[paper_id] = metrics
        return metrics
    
    fn consolidate_reviews(self, paper_id: String) -> Dict[String, String]:
        """Consolidates all reviews for a paper into final decision"""
        var decision = Dict[String, String]()
        
        if paper_id not in self.active_reviews:
            decision["status"] = "no_reviews"
            decision["recommendation"] = "awaiting_review"
            return decision
        
        var reviews = self.active_reviews[paper_id]
        var recommendations = Dict[String, Int]()
        
        # Count recommendations
        for review in reviews:
            if review.overall_recommendation in recommendations:
                recommendations[review.overall_recommendation] += 1
            else:
                recommendations[review.overall_recommendation] = 1
        
        # Determine consensus
        var accept_count = recommendations.get("accept", 0) + recommendations.get("minor_revision", 0)
        var reject_count = recommendations.get("reject", 0)
        var major_revision_count = recommendations.get("major_revision", 0)
        
        if reject_count > 0:
            decision["recommendation"] = "reject"
        elif major_revision_count > accept_count:
            decision["recommendation"] = "major_revision"
        elif accept_count >= len(reviews) // 2:
            decision["recommendation"] = "accept"
        else:
            decision["recommendation"] = "minor_revision"
        
        decision["status"] = "review_complete"
        decision["review_count"] = str(len(reviews))
        
        return decision

fn create_cognitive_science_validation_system() -> ValidationSystem:
    """Creates validation system specialized for cognitive science"""
    var system = ValidationSystem()
    
    # Add sample reviewers with cognitive science expertise
    var reviewer1 = PeerReviewer(
        "rev001",
        "Dr. Sarah Chen",
        List[String]("mind-wandering", "attention", "fMRI", "cognitive neuroscience"),
        "Stanford University"
    )
    reviewer1.h_index = 35
    system.reviewer_pool["rev001"] = reviewer1
    
    var reviewer2 = PeerReviewer(
        "rev002",
        "Prof. Michael Johnson",
        List[String]("consciousness", "metacognition", "theoretical neuroscience"),
        "MIT"
    )
    reviewer2.h_index = 42
    system.reviewer_pool["rev002"] = reviewer2
    
    var reviewer3 = PeerReviewer(
        "rev003",
        "Dr. Elena Rodriguez",
        List[String]("computational modeling", "recursive processes", "complex systems"),
        "UC Berkeley"
    )
    reviewer3.h_index = 28
    system.reviewer_pool["rev003"] = reviewer3
    
    return system