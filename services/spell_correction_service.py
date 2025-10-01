"""
Spell Correction Service - Intelligent Text Correction
======================================================
This module provides multi-library spell correction with automatic pattern detection,
domain-specific learning, and confidence scoring for robust query correction.
"""

import re
import time
import logging
from typing import List, Dict, Tuple, Optional
from difflib import get_close_matches
from collections import Counter
import json
import os

try:
    from spellchecker import SpellChecker
    PYSPELLCHECKER_AVAILABLE = True
except ImportError:
    PYSPELLCHECKER_AVAILABLE = False

try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    from autocorrect import Speller
    AUTOCORRECT_AVAILABLE = True
except ImportError:
    AUTOCORRECT_AVAILABLE = False


class SpellCorrectionService:
    """
    Multi-library spell correction service with domain adaptation.
    
    Combines multiple spell checking libraries (pyspellchecker, enchant, autocorrect)
    with custom dictionary learning for domain-specific terminology.
    Supports automatic pattern correction and confidence scoring.
    """
    
    def __init__(self, config_service, cache_manager, language='en'):
        """
        Initialize the spell correction service.
        
        Args:
            config_service: Configuration service instance
            cache_manager: Cache manager instance
            language: Language code (default: 'en')
        """
        self.config_service = config_service
        self.cache_manager = cache_manager
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        self.word_frequency = Counter()
        self.custom_dictionary = set()
        
        self.spell_checkers = {}
        self.automatic_patterns = self._get_automatic_patterns()
        
        self.is_available = self._initialize_spell_checkers()
        
        if self.is_available:
            self.logger.info(f"Spell correction service initialized with {len(self.spell_checkers)} checkers")
        else:
            self.logger.warning("Spell correction service unavailable - no spell checkers found")
    
    def _initialize_spell_checkers(self) -> bool:
        """
        Initialize available spell checking libraries.
        
        Returns:
            bool: True if at least one checker loaded successfully
        """
        checkers_loaded = 0
        
        if PYSPELLCHECKER_AVAILABLE:
            try:
                self.spell_checkers['pyspellchecker'] = SpellChecker(language=self.language)
                checkers_loaded += 1
                self.logger.info("PySpellChecker loaded")
            except Exception as e:
                self.logger.warning(f"PySpellChecker initialization failed: {e}")
        
        if ENCHANT_AVAILABLE:
            try:
                self.spell_checkers['enchant'] = enchant.Dict(self.language)
                checkers_loaded += 1
                self.logger.info("PyEnchant loaded")
            except Exception as e:
                self.logger.warning(f"PyEnchant initialization failed: {e}")
        
        if AUTOCORRECT_AVAILABLE:
            try:
                self.spell_checkers['autocorrect'] = Speller(lang=self.language)
                checkers_loaded += 1
                self.logger.info("Autocorrect loaded")
            except Exception as e:
                self.logger.warning(f"Autocorrect initialization failed: {e}")
        
        if checkers_loaded == 0:
            self.logger.warning("No spell checkers available!")
            return False
        
        return True
    
    def _get_automatic_patterns(self) -> Dict[str, str]:
        """
        Get regex patterns for automatic text corrections.
        
        Returns:
            Dict[str, str]: Pattern to replacement mapping
        """
        return {
            r'(.)\1{3,}': r'\1\1',  # Reduce repeated characters (3+ to 2)
            r'\s+': ' ',  # Normalize whitespace
            r'([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])\1{2,}': r'\1',  # Repeated consonants
        }
    
    def add_to_dictionary(self, words: List[str]) -> None:
        """
        Add words to custom dictionary.
        
        Args:
            words: List of words to add
        """
        if not self.is_available:
            return
        
        self.custom_dictionary.update(word.lower() for word in words)
        self.logger.debug(f"Added {len(words)} words to custom dictionary")
    
    def learn_from_text(self, text: str) -> None:
        """
        Learn domain-specific vocabulary from text.
        
        Analyzes text to identify frequently occurring words and
        automatically adds them to the custom dictionary.
        
        Args:
            text: Training text containing domain vocabulary
        """
        if not self.is_available:
            return
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        self.word_frequency.update(words)
        
        # Add frequently occurring words to dictionary
        frequent_words = [word for word, count in self.word_frequency.items() 
                         if count >= 3 and len(word) > 4]
        self.custom_dictionary.update(frequent_words)
        
        if frequent_words:
            self.logger.info(f"Auto-learned {len(frequent_words)} domain terms from text")
    
    def is_correct(self, word: str) -> bool:
        """
        Check if a word is spelled correctly.
        
        Args:
            word: Word to check
            
        Returns:
            bool: True if word is correct
        """
        if not self.is_available:
            return True
        
        word = word.lower().strip()
        
        # Check custom dictionary first
        if word in self.custom_dictionary:
            return True
        
        # Skip short words and numbers
        if len(word) <= 2 or word.isdigit():
            return True
        
        # Check with available spell checkers
        for checker_name, checker in self.spell_checkers.items():
            try:
                if checker_name == 'pyspellchecker':
                    if word in checker:
                        return True
                elif checker_name == 'enchant':
                    if checker.check(word):
                        return True
                elif checker_name == 'autocorrect':
                    corrected = checker(word)
                    if corrected.lower() == word.lower():
                        return True
            except:
                continue
        
        return False
    
    def get_suggestions(self, word: str, max_suggestions: int = 5) -> List[str]:
        """
        Get spelling suggestions for a word.
        
        Args:
            word: Misspelled word
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List[str]: List of suggested corrections
        """
        if not self.is_available:
            return [word]
        
        word = word.lower().strip()
        
        if self.is_correct(word):
            return [word]
        
        all_suggestions = set()
        
        # Get suggestions from each checker
        for checker_name, checker in self.spell_checkers.items():
            try:
                if checker_name == 'pyspellchecker':
                    suggestions = list(checker.candidates(word))
                    all_suggestions.update(suggestions[:3])
                    
                elif checker_name == 'enchant':
                    suggestions = checker.suggest(word)
                    all_suggestions.update(suggestions[:3])
                    
                elif checker_name == 'autocorrect':
                    corrected = checker(word)
                    if corrected.lower() != word.lower():
                        all_suggestions.add(corrected)
                        
            except Exception:
                continue
        
        # Add suggestions from learned vocabulary
        if self.word_frequency:
            vocab_list = [w for w in self.word_frequency.keys() if len(w) > 3]
            difflib_suggestions = get_close_matches(
                word, vocab_list, n=3, cutoff=0.7
            )
            all_suggestions.update(difflib_suggestions)
        
        suggestion_list = list(all_suggestions)
        
        # Sort by frequency if available
        if self.word_frequency:
            suggestion_list.sort(
                key=lambda w: self.word_frequency.get(w.lower(), 0), 
                reverse=True
            )
        
        return suggestion_list[:max_suggestions]
    
    def _apply_automatic_patterns(self, word: str) -> List[str]:
        """
        Apply automatic correction patterns to a word.
        
        Args:
            word: Word to correct
            
        Returns:
            List[str]: List of pattern-corrected words
        """
        suggestions = []
        
        for pattern, replacement in self.automatic_patterns.items():
            try:
                corrected = re.sub(pattern, replacement, word)
                if corrected != word and len(corrected) > 1:
                    if self.is_correct(corrected):
                        suggestions.append(corrected)
            except:
                continue
        
        return suggestions
    
    def correct_word(self, word: str, confidence_threshold: float = 0.8) -> Tuple[str, float]:
        """
        Correct a single word with confidence scoring.
        
        Args:
            word: Word to correct
            confidence_threshold: Minimum confidence to apply correction
            
        Returns:
            Tuple[str, float]: Corrected word and confidence score
        """
        if not self.is_available:
            return word, 1.0
        
        if self.is_correct(word):
            return word, 1.0
        
        # Try automatic pattern corrections first
        pattern_suggestions = self._apply_automatic_patterns(word)
        if pattern_suggestions:
            return pattern_suggestions[0], 0.9
        
        # Get suggestions from spell checkers
        suggestions = self.get_suggestions(word)
        
        if not suggestions:
            return word, 0.0
        
        best_suggestion = suggestions[0]
        confidence = self._calculate_confidence(word, best_suggestion)
        
        if confidence >= confidence_threshold:
            return best_suggestion, confidence
        else:
            return word, confidence
    
    def _calculate_confidence(self, original: str, suggestion: str) -> float:
        """
        Calculate confidence score for a spelling correction.
        
        Args:
            original: Original word
            suggestion: Suggested correction
            
        Returns:
            float: Confidence score (0-1)
        """
        if not FUZZYWUZZY_AVAILABLE:
            # Fallback to Levenshtein distance
            max_len = max(len(original), len(suggestion))
            if max_len == 0:
                return 1.0
            
            edit_distance = self._levenshtein_distance(original, suggestion)
            return max(0.0, 1.0 - (edit_distance / max_len))
        
        return fuzz.ratio(original, suggestion) / 100.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein (edit) distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            int: Edit distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def correct_text(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Correct spelling in an entire text string.
        
        Preserves original capitalization and only corrects words
        that meet the confidence threshold.
        
        Args:
            text: Text to correct
            confidence_threshold: Minimum confidence for corrections
            
        Returns:
            Dict: Dictionary with original text, corrected text, and corrections list
        """
        if not self.is_available:
            return {
                'original_text': text,
                'corrected_text': text,
                'corrections': [],
                'has_corrections': False,
                'correction_count': 0
            }
        
        # Apply automatic pattern corrections first
        cleaned_text = text
        for pattern, replacement in self.automatic_patterns.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Find all words with their positions
        word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        words_with_positions = []
        
        for match in word_pattern.finditer(cleaned_text):
            words_with_positions.append({
                'word': match.group(),
                'start': match.start(),
                'end': match.end(),
                'original_case': match.group()
            })
        
        corrections = []
        corrected_text = cleaned_text
        offset = 0
        
        # Correct each word
        for word_info in words_with_positions:
            word = word_info['word'].lower()
            original_case = word_info['original_case']
            
            # Skip very short words
            if len(word) <= 2:
                continue
            
            corrected_word, confidence = self.correct_word(word, confidence_threshold)
            
            if corrected_word != word and confidence >= confidence_threshold:
                # Preserve original capitalization
                if original_case.isupper():
                    corrected_word = corrected_word.upper()
                elif original_case.istitle():
                    corrected_word = corrected_word.capitalize()
                
                start_pos = word_info['start'] + offset
                end_pos = word_info['end'] + offset
                
                corrected_text = (corrected_text[:start_pos] + 
                                corrected_word + 
                                corrected_text[end_pos:])
                
                offset += len(corrected_word) - len(original_case)
                
                corrections.append({
                    'original': original_case,
                    'corrected': corrected_word,
                    'confidence': confidence,
                    'position': word_info['start'],
                    'method': 'automatic'
                })
        
        return {
            'original_text': text,
            'corrected_text': corrected_text,
            'corrections': corrections,
            'has_corrections': len(corrections) > 0,
            'correction_count': len(corrections)
        }
    
    def get_service_stats(self) -> Dict:
        """
        Get service statistics and availability info.
        
        Returns:
            Dict: Service statistics
        """
        return {
            'is_available': self.is_available,
            'available_checkers': list(self.spell_checkers.keys()),
            'custom_dictionary_size': len(self.custom_dictionary),
            'learned_vocabulary_size': len(self.word_frequency),
            'automatic_patterns': len(self.automatic_patterns),
            'libraries_available': {
                'pyspellchecker': PYSPELLCHECKER_AVAILABLE,
                'enchant': ENCHANT_AVAILABLE,
                'fuzzywuzzy': FUZZYWUZZY_AVAILABLE,
                'autocorrect': AUTOCORRECT_AVAILABLE
            }
        }
    
    def save_dictionary(self, filepath: str) -> None:
        """
        Save learned dictionary to file.
        
        Args:
            filepath: Path to save dictionary JSON
        """
        if not self.is_available:
            return
        
        data = {
            'custom_dictionary': list(self.custom_dictionary),
            'word_frequency': dict(self.word_frequency.most_common(10000)),
            'creation_method': 'automatic_learning_only',
            'created_at': time.time()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Dictionary saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save dictionary: {e}")
    
    def load_dictionary(self, filepath: str) -> None:
        """
        Load learned dictionary from file.
        
        Args:
            filepath: Path to dictionary JSON file
        """
        if not self.is_available or not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.custom_dictionary = set(data.get('custom_dictionary', []))
            self.word_frequency = Counter(data.get('word_frequency', {}))
            
            self.logger.info(f"Dictionary loaded: {len(self.custom_dictionary)} learned words")
        except Exception as e:
            self.logger.error(f"Error loading dictionary: {e}")