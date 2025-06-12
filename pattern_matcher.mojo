from algorithm import vectorize
from tensor import Tensor
from collections import Dict, List
from utils.hash import hash
import math

struct ResearchPattern:
    var pattern_id: String
    var pattern_type: String
    var keywords: List[String]
    var weight: Float32
    var context_window: Int
    
    fn __init__(inout self, pattern_id: String, pattern_type: String, 
                keywords: List[String], weight: Float32 = 1.0, 
                context_window: Int = 100):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        self.keywords = keywords
        self.weight = weight
        self.context_window = context_window

struct PatternMatch:
    var pattern_id: String
    var confidence: Float32
    var location: Int
    var context: String
    var matched_keywords: List[String]
    
    fn __init__(inout self, pattern_id: String, confidence: Float32, 
                location: Int, context: String):
        self.pattern_id = pattern_id
        self.confidence = confidence
        self.location = location
        self.context = context
        self.matched_keywords = List[String]()

struct ResearchSignature:
    var author_name: String
    var signature_patterns: List[ResearchPattern]
    var minimum_confidence: Float32
    var combination_rules: Dict[String, List[String]]
    
    fn __init__(inout self, author_name: String, min_confidence: Float32 = 0.7):
        self.author_name = author_name
        self.signature_patterns = List[ResearchPattern]()
        self.minimum_confidence = min_confidence
        self.combination_rules = Dict[String, List[String]]()

struct PatternMatcher:
    var signatures: Dict[String, ResearchSignature]
    var pattern_cache: Dict[String, List[PatternMatch]]
    var text_embeddings: Dict[String, Tensor[DType.float32]]
    
    fn __init__(inout self):
        self.signatures = Dict[String, ResearchSignature]()
        self.pattern_cache = Dict[String, List[PatternMatch]]()
        self.text_embeddings = Dict[String, Tensor[DType.float32]]()
    
    fn create_oates_signature(inout self) -> ResearchSignature:
        """Creates research signature for Oates R's work"""
        var signature = ResearchSignature("Oates, R.", 0.75)
        
        # Mind-wandering patterns
        var mw_pattern = ResearchPattern(
            "mind_wandering_core",
            "conceptual",
            List[String]("mind-wandering", "task-unrelated thought", 
                        "spontaneous thought", "default mode network"),
            weight=2.0,
            context_window=150
        )
        signature.signature_patterns.append(mw_pattern)
        
        # Recursive cognition patterns
        var recursive_pattern = ResearchPattern(
            "recursive_cognition",
            "theoretical",
            List[String]("recursive", "self-referential", "meta-awareness",
                        "hierarchical processing", "nested loops"),
            weight=2.5,
            context_window=200
        )
        signature.signature_patterns.append(recursive_pattern)
        
        # Attention dynamics patterns
        var attention_pattern = ResearchPattern(
            "attention_dynamics",
            "methodological",
            List[String]("attention fluctuations", "attentional state",
                        "sustained attention", "attention regulation",
                        "attention-recognition decoupling"),
            weight=2.0,
            context_window=150
        )
        signature.signature_patterns.append(attention_pattern)
        
        # Fractal/scale-invariant patterns
        var fractal_pattern = ResearchPattern(
            "fractal_dynamics",
            "analytical",
            List[String]("fractal", "scale-invariant", "power law",
                        "self-similar", "multiscale"),
            weight=1.8,
            context_window=100
        )
        signature.signature_patterns.append(fractal_pattern)
        
        # DMN patterns
        var dmn_pattern = ResearchPattern(
            "dmn_dynamics",
            "neuroscientific",
            List[String]("default mode", "DMN", "resting state",
                        "intrinsic networks", "anticorrelation"),
            weight=1.5,
            context_window=150
        )
        signature.signature_patterns.append(dmn_pattern)
        
        # Define combination rules for higher confidence
        signature.combination_rules["high_confidence"] = List[String](
            "mind_wandering_core+recursive_cognition",
            "attention_dynamics+fractal_dynamics",
            "dmn_dynamics+mind_wandering_core"
        )
        
        signature.combination_rules["very_high_confidence"] = List[String](
            "mind_wandering_core+recursive_cognition+attention_dynamics",
            "fractal_dynamics+dmn_dynamics+recursive_cognition"
        )
        
        return signature
    
    fn match_patterns(self, text: String, signature: ResearchSignature) -> List[PatternMatch]:
        """Matches patterns in text against research signature"""
        var matches = List[PatternMatch]()
        var text_lower = text.lower()
        
        for pattern in signature.signature_patterns:
            var pattern_matches = self._find_pattern_matches(text_lower, pattern)
            matches.extend(pattern_matches)
        
        return matches
    
    fn _find_pattern_matches(self, text: String, pattern: ResearchPattern) -> List[PatternMatch]:
        """Finds all matches for a specific pattern in text"""
        var matches = List[PatternMatch]()
        
        for keyword in pattern.keywords:
            var position = 0
            while True:
                var found = text.find(keyword, position)
                if found == -1:
                    break
                
                # Extract context around match
                var start = max(0, found - pattern.context_window)
                var end = min(len(text), found + len(keyword) + pattern.context_window)
                var context = text[start:end]
                
                # Calculate confidence based on keyword relevance and context
                var confidence = self._calculate_match_confidence(
                    keyword, context, pattern
                )
                
                var match = PatternMatch(
                    pattern.pattern_id,
                    confidence * pattern.weight,
                    found,
                    context
                )
                match.matched_keywords.append(keyword)
                matches.append(match)
                
                position = found + 1
        
        return matches
    
    fn _calculate_match_confidence(self, keyword: String, context: String, 
                                 pattern: ResearchPattern) -> Float32:
        """Calculates confidence score for a pattern match"""
        var base_confidence: Float32 = 0.5
        
        # Boost confidence if multiple keywords from same pattern appear in context
        var keyword_count = 0
        for kw in pattern.keywords:
            if kw in context.lower():
                keyword_count += 1
        
        if keyword_count > 1:
            base_confidence += 0.1 * Float32(keyword_count - 1)
        
        # Boost for academic context indicators
        var academic_indicators = ["study", "research", "analysis", "findings",
                                 "results", "methodology", "hypothesis", "data"]
        for indicator in academic_indicators:
            if indicator in context.lower():
                base_confidence += 0.05
        
        # Cap confidence at 1.0
        return min(base_confidence, 1.0)
    
    fn calculate_aggregate_confidence(self, matches: List[PatternMatch], 
                                    signature: ResearchSignature) -> Float32:
        """Calculates aggregate confidence from all pattern matches"""
        if len(matches) == 0:
            return 0.0
        
        # Group matches by pattern
        var pattern_scores = Dict[String, Float32]()
        for match in matches:
            if match.pattern_id not in pattern_scores:
                pattern_scores[match.pattern_id] = 0.0
            pattern_scores[match.pattern_id] = max(
                pattern_scores[match.pattern_id], 
                match.confidence
            )
        
        # Check combination rules for bonus confidence
        var combination_bonus: Float32 = 0.0
        
        for rule_type, rules in signature.combination_rules.items():
            for rule in rules:
                var required_patterns = rule.split("+")
                var all_present = True
                
                for required in required_patterns:
                    if required not in pattern_scores or pattern_scores[required] < 0.5:
                        all_present = False
                        break
                
                if all_present:
                    if rule_type == "high_confidence":
                        combination_bonus = max(combination_bonus, 0.2)
                    elif rule_type == "very_high_confidence":
                        combination_bonus = max(combination_bonus, 0.3)
        
        # Calculate weighted average of pattern scores
        var total_score: Float32 = 0.0
        var total_weight: Float32 = 0.0
        
        for pattern_id, score in pattern_scores.items():
            total_score += score
            total_weight += 1.0
        
        if total_weight > 0:
            var avg_score = total_score / total_weight
            return min(avg_score + combination_bonus, 1.0)
        
        return 0.0
    
    fn extract_research_metadata(self, text: String, matches: List[PatternMatch]) -> Dict[String, String]:
        """Extracts research metadata based on pattern matches"""
        var metadata = Dict[String, String]()
        
        # Extract title (usually first substantial line)
        var lines = text.split('\n')
        for line in lines:
            if len(line.strip()) > 10:
                metadata["title"] = line.strip()
                break
        
        # Identify research domain based on matched patterns
        var domain_indicators = Dict[String, Int]()
        for match in matches:
            if "mind_wandering" in match.pattern_id:
                domain_indicators["cognitive_neuroscience"] = domain_indicators.get("cognitive_neuroscience", 0) + 1
            elif "recursive" in match.pattern_id:
                domain_indicators["computational_cognition"] = domain_indicators.get("computational_cognition", 0) + 1
            elif "dmn" in match.pattern_id:
                domain_indicators["neuroscience"] = domain_indicators.get("neuroscience", 0) + 1
        
        # Set primary domain
        var max_count = 0
        var primary_domain = "general_cognitive_science"
        for domain, count in domain_indicators.items():
            if count > max_count:
                max_count = count
                primary_domain = domain
        
        metadata["domain"] = primary_domain
        
        # Extract methodology hints
        var methodology_keywords = ["fmri", "eeg", "behavioral", "computational",
                                  "meta-analysis", "experimental", "longitudinal"]
        var found_methods = List[String]()
        for keyword in methodology_keywords:
            if keyword in text.lower():
                found_methods.append(keyword)
        
        if len(found_methods) > 0:
            metadata["methodology_hints"] = ",".join(found_methods)
        
        return metadata
    
    fn identify_author_contribution(self, text: String, author_name: String) -> Bool:
        """Identifies if the author contributed to the research"""
        var text_lower = text.lower()
        var author_lower = author_name.lower()
        
        # Direct author mention
        if author_lower in text_lower:
            return True
        
        # Check for variations (last name only, initials, etc.)
        var name_parts = author_name.split(",")
        if len(name_parts) > 0:
            var last_name = name_parts[0].strip().lower()
            if last_name in text_lower:
                # Additional verification for common names
                if self._verify_author_context(text_lower, last_name):
                    return True
        
        return False
    
    fn _verify_author_context(self, text: String, author_name: String) -> Bool:
        """Verifies author mention is in appropriate context"""
        var author_pos = text.find(author_name)
        if author_pos == -1:
            return False
        
        # Check surrounding context
        var start = max(0, author_pos - 50)
        var end = min(len(text), author_pos + 50)
        var context = text[start:end]
        
        var author_indicators = ["author", "by", "et al", "and", "correspondence",
                               "affiliation", "email", "orcid"]
        
        for indicator in author_indicators:
            if indicator in context:
                return True
        
        return False

fn create_default_pattern_matcher() -> PatternMatcher:
    """Creates pattern matcher with Oates signature pre-loaded"""
    var matcher = PatternMatcher()
    var oates_signature = matcher.create_oates_signature()
    matcher.signatures["oates_r"] = oates_signature
    return matcher