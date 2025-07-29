from typing import Dict, Any, List
import toml
import os
from pathlib import Path

class PrivacyScorer:
    def __init__(self, scoring_rules=None):
        if scoring_rules is None:
            self.rules = self.load_default_rules()
        else:
            self.rules = scoring_rules
        
        # Load detailed scoring criteria
        self.criteria = self.load_scoring_criteria()
            
    def load_default_rules(self) -> Dict:
        """Load default scoring rules from config file"""
        rules_path = Path("privacy/config/scoring_rules.toml")
        if not rules_path.exists():
            raise FileNotFoundError(f"Scoring rules not found at: {rules_path}")
        return toml.load(rules_path)
    
    def load_scoring_criteria(self) -> Dict:
        """Load detailed scoring criteria"""
        criteria_path = Path("privacy_scorer/criteria/scoring_criteria.toml")
        if not criteria_path.exists():
            raise FileNotFoundError(f"Scoring criteria not found at: {criteria_path}")
        return toml.load(criteria_path)
            
    def score_policy(self, policy_text: str) -> Dict[str, Any]:
        """Score the entire privacy policy"""
        if not policy_text:
            return self._create_error_result("Empty policy text")
        
        try:
            total_score = 0.0
            category_scores = {}
            
            # Score each category
            for category in self.criteria['categories'].keys():
                result = self._score_category(category, policy_text)
                category_scores[category] = result
                total_score += result['weighted_score']
            
            grade = self._calculate_grade(total_score)
            
            return {
                "total_score": round(total_score, 1),
                "grade": grade,
                "category_scores": category_scores,
                "feedback": [score["feedback"] for score in category_scores.values()],
                "summary": self._generate_summary(total_score, grade)
            }
        
        except Exception as e:
            return self._create_error_result(f"Error scoring policy: {str(e)}")

    def _check_phrases(self, text: str, phrases: list) -> tuple[bool, list[str]]:
        """
        Check for presence of phrases in text and return matching phrases
        """
        text = text.lower()
        found_phrases = [phrase for phrase in phrases if phrase.lower() in text]
        return bool(found_phrases), found_phrases

    def _parse_item(self, text: str) -> tuple[str, Any]:
        """Parse a single item from the policy text"""
        try:
            # Basic parsing logic - can be expanded based on needs
            parts = text.split(':', 1)
            if len(parts) < 2:
                return None  # Return None if we can't parse properly
            
            key = parts[0].strip()
            value = parts[1].strip()
            return (key, value)
        except Exception:
            return None

    def _score_category(self, category: str, policy_text: str) -> Dict[str, Any]:
        """Score a specific category of the privacy policy"""
        raw_score = 0.0
        feedback = []
        
        try:
            # Get the criteria for this category
            category_criteria = self.criteria['categories'].get(category, [])
            
            if not category_criteria:
                return {
                    "raw_score": 0.0,
                    "weighted_score": 0.0,
                    "feedback": [f"No criteria defined for category: {category}"]
                }
            
            for criterion in category_criteria:
                criterion_data = self.criteria['criteria'].get(criterion, {})
                if not criterion_data:
                    continue
                
                # Get phrases
                required_phrases = criterion_data.get('required_phrases', [])
                matching_phrases = criterion_data.get('matching_phrases', [])
                
                # Check phrases
                required_found, found_required = self._check_phrases(policy_text, required_phrases)
                matching_found, found_matching = self._check_phrases(policy_text, matching_phrases)
                
                points = float(criterion_data.get('points', 0))
                
                if required_found:
                    if matching_found:
                        raw_score += points
                    else:
                        raw_score += points / 2
                        feedback.append(f"Basic {criterion} requirements met, but could be more specific")
                else:
                    feedback.append(f"No clear {criterion} information found")

            # Calculate weighted score
            weight = float(self.criteria['scoring_weights'].get(category, 1.0))  # Default weight of 1.0
            max_possible = len(category_criteria) * 100.0
            normalized_score = (raw_score / max_possible * 100.0) if max_possible > 0 else 0.0
            weighted_score = (normalized_score / 100.0) * weight

            return {
                "raw_score": round(raw_score, 2),
                "weighted_score": round(weighted_score, 2),
                "feedback": feedback
            }
        
        except Exception as e:
            return {
                "raw_score": 0.0,
                "weighted_score": 0.0,
                "feedback": [f"Error scoring category {category}: {str(e)}"]
            }

    def _calculate_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 50:
            return "C"
        elif score >= 25:
            return "D"
        else:
            return "F"

    def _generate_detailed_feedback(self, category_scores: Dict[str, Any]) -> List[str]:
        """Generate detailed feedback based on category scores and findings"""
        detailed_feedback = []
        
        for category, scores in category_scores.items():
            score = scores['raw_score']
            category_name = category.replace('_', ' ').title()
            
            if score < 50:
                detailed_feedback.append(
                    f"Critical: {category_name} needs significant improvement. "
                    f"Review the requirements and enhance the policy accordingly."
                )
            elif score < 75:
                detailed_feedback.append(
                    f"{category_name} meets basic requirements but could be more comprehensive."
                )
            elif score < 90:
                detailed_feedback.append(
                    f"{category_name} policies are good but could be enhanced with more specific details."
                )

            # Add specific feedback from scoring
            detailed_feedback.extend(scores['feedback'])

        return detailed_feedback

    def _generate_summary(self, score: float, grade: str) -> str:
        """Generate an overall summary based on the score"""
        if score >= 90:
            return "Excellent privacy policy with comprehensive coverage of all major areas."
        elif score >= 75:
            return "Good privacy policy with room for minor improvements in specific areas."
        elif score >= 50:
            return "Adequate privacy policy but needs enhancement in several areas."
        elif score >= 25:
            return "Privacy policy needs significant improvement to meet best practices."
        else:
            return "Privacy policy requires major revision to address fundamental requirements."

    def _create_error_result(self, error_message: str) -> Dict:
        """Create an error result dictionary"""
        return {
            "total_score": 0.0,
            "grade": "F",
            "category_scores": {
                category: {
                    "raw_score": 0.0,
                    "weighted_score": 0.0,
                    "feedback": [error_message]
                } for category in self.criteria['categories'].keys()
            },
            "feedback": [error_message],
            "summary": "Unable to evaluate privacy policy"
        }
