"""
MATHTOKENIZER 2.0
=================

Full tokenizer replacement for mathematical text.

Problems with standard tokenizers:
- Split "123456" into multiple tokens
- Don't understand LaTeX commands
- Treat "π" same as random unicode
- No equation structure awareness

MathTokenizer fixes this:
1. Number-aware: keeps integers/floats as single tokens
2. LaTeX-native: proper handling of \frac, \sum, etc.
3. Operator-aware: math operators get dedicated tokens
4. Structure preservation: equation delimiters tracked
5. Fast C-style parsing (pure Python for now, Cython later)

Target: 2x faster encoding, better downstream accuracy.

Author: Ryan J Cardwell (Archer Phoenix)
"""

import re
from typing import Dict, List, Optional, Tuple, Set, Iterator, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict
import json
import struct
from pathlib import Path


# =============================================================================
# TOKEN TYPES
# =============================================================================

class TokenType(Enum):
    """Types of tokens in mathematical text."""
    # Special
    PAD = auto()
    UNK = auto()
    BOS = auto()
    EOS = auto()
    
    # Numbers
    INTEGER = auto()
    FLOAT = auto()
    FRACTION = auto()
    SCIENTIFIC = auto()
    
    # Math operators
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    DIVIDE = auto()
    EQUALS = auto()
    LESS = auto()
    GREATER = auto()
    LEQ = auto()
    GEQ = auto()
    NEQ = auto()
    APPROX = auto()
    
    # Grouping
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    
    # LaTeX
    LATEX_CMD = auto()
    LATEX_ENV_BEGIN = auto()
    LATEX_ENV_END = auto()
    LATEX_FRAC = auto()
    LATEX_SQRT = auto()
    LATEX_SUM = auto()
    LATEX_PROD = auto()
    LATEX_INT = auto()
    LATEX_LIM = auto()
    
    # Greek letters
    GREEK = auto()
    
    # Variables
    VARIABLE = auto()
    
    # Text
    WORD = auto()
    PUNCTUATION = auto()
    WHITESPACE = auto()
    NEWLINE = auto()


# =============================================================================
# TOKEN DEFINITION
# =============================================================================

@dataclass
class Token:
    """A single token."""
    text: str
    token_type: TokenType
    token_id: int = -1
    
    # For numbers
    numeric_value: Optional[float] = None
    
    # Position in original text
    start: int = 0
    end: int = 0


# =============================================================================
# LEXER
# =============================================================================

class MathLexer:
    """
    Lexer for mathematical text.
    
    Converts raw text into token stream.
    """
    
    # Regex patterns (order matters!)
    PATTERNS = [
        # Scientific notation (before float/int)
        (r'-?\d+\.?\d*[eE][+-]?\d+', TokenType.SCIENTIFIC),
        
        # Fractions like 3/4
        (r'\d+/\d+', TokenType.FRACTION),
        
        # Floats
        (r'-?\d+\.\d+', TokenType.FLOAT),
        
        # Integers
        (r'-?\d+', TokenType.INTEGER),
        
        # LaTeX commands
        (r'\\frac\b', TokenType.LATEX_FRAC),
        (r'\\sqrt\b', TokenType.LATEX_SQRT),
        (r'\\sum\b', TokenType.LATEX_SUM),
        (r'\\prod\b', TokenType.LATEX_PROD),
        (r'\\int\b', TokenType.LATEX_INT),
        (r'\\lim\b', TokenType.LATEX_LIM),
        (r'\\begin\{[^}]+\}', TokenType.LATEX_ENV_BEGIN),
        (r'\\end\{[^}]+\}', TokenType.LATEX_ENV_END),
        (r'\\[a-zA-Z]+', TokenType.LATEX_CMD),
        
        # Greek letters (common ones)
        (r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', TokenType.GREEK),
        
        # Operators
        (r'\+', TokenType.PLUS),
        (r'-', TokenType.MINUS),
        (r'\*|×|·', TokenType.TIMES),
        (r'/|÷', TokenType.DIVIDE),
        (r'=', TokenType.EQUALS),
        (r'≤|<=', TokenType.LEQ),
        (r'≥|>=', TokenType.GEQ),
        (r'≠|!=', TokenType.NEQ),
        (r'≈|~', TokenType.APPROX),
        (r'<', TokenType.LESS),
        (r'>', TokenType.GREATER),
        
        # Grouping
        (r'\(', TokenType.LPAREN),
        (r'\)', TokenType.RPAREN),
        (r'\[', TokenType.LBRACKET),
        (r'\]', TokenType.RBRACKET),
        (r'\{', TokenType.LBRACE),
        (r'\}', TokenType.RBRACE),
        
        # Variables (single letters)
        (r'[a-zA-Z]', TokenType.VARIABLE),
        
        # Words
        (r'[a-zA-Z][a-zA-Z0-9]*', TokenType.WORD),
        
        # Punctuation
        (r'[.,;:!?\'"()]', TokenType.PUNCTUATION),
        
        # Whitespace
        (r'[ \t]+', TokenType.WHITESPACE),
        (r'\n', TokenType.NEWLINE),
    ]
    
    def __init__(self):
        # Compile patterns
        self.compiled = [(re.compile(p), t) for p, t in self.PATTERNS]
    
    def tokenize(self, text: str) -> Iterator[Token]:
        """Tokenize text into token stream."""
        pos = 0
        
        while pos < len(text):
            match = None
            token_type = None
            
            # Try each pattern
            for pattern, ttype in self.compiled:
                m = pattern.match(text, pos)
                if m:
                    if match is None or len(m.group()) > len(match.group()):
                        match = m
                        token_type = ttype
            
            if match:
                token_text = match.group()
                
                # Extract numeric value for numbers
                numeric_value = None
                if token_type in (TokenType.INTEGER, TokenType.FLOAT, TokenType.SCIENTIFIC):
                    try:
                        numeric_value = float(token_text)
                    except ValueError:
                        pass
                elif token_type == TokenType.FRACTION:
                    parts = token_text.split('/')
                    if len(parts) == 2:
                        try:
                            numeric_value = float(parts[0]) / float(parts[1])
                        except (ValueError, ZeroDivisionError):
                            pass
                
                yield Token(
                    text=token_text,
                    token_type=token_type,
                    numeric_value=numeric_value,
                    start=pos,
                    end=pos + len(token_text),
                )
                
                pos = match.end()
            else:
                # Unknown character - emit as UNK
                yield Token(
                    text=text[pos],
                    token_type=TokenType.UNK,
                    start=pos,
                    end=pos + 1,
                )
                pos += 1


# =============================================================================
# VOCABULARY
# =============================================================================

class MathVocab:
    """
    Vocabulary for math tokenizer.
    
    Structure:
    - Special tokens: 0-99
    - Numbers: 100-999 (encoded specially)
    - Operators: 1000-1099
    - Greek: 1100-1199
    - LaTeX commands: 1200-1499
    - Common words: 1500+
    """
    
    # Reserved ranges
    SPECIAL_START = 0
    NUMBER_START = 100
    OPERATOR_START = 1000
    GREEK_START = 1100
    LATEX_START = 1200
    WORD_START = 1500
    
    # Special tokens
    SPECIAL_TOKENS = {
        '<pad>': 0,
        '<unk>': 1,
        '<bos>': 2,
        '<eos>': 3,
        '<num>': 4,      # Number placeholder
        '<var>': 5,      # Variable placeholder
        '<eq>': 6,       # Equation marker
        '<proof>': 7,    # Proof marker
        '<qed>': 8,      # QED marker
        '<boxed>': 9,    # Boxed answer
    }
    
    # Operators
    OPERATORS = {
        '+': 1000, '-': 1001, '*': 1002, '/': 1003, '=': 1004,
        '<': 1005, '>': 1006, '≤': 1007, '≥': 1008, '≠': 1009,
        '≈': 1010, '(': 1011, ')': 1012, '[': 1013, ']': 1014,
        '{': 1015, '}': 1016, '^': 1017, '_': 1018, '!': 1019,
        '%': 1020, '&': 1021, '|': 1022, ',': 1023, '.': 1024,
        ':': 1025, ';': 1026, '?': 1027, '@': 1028, '#': 1029,
    }
    
    # Greek letters
    GREEK = {
        'α': 1100, 'β': 1101, 'γ': 1102, 'δ': 1103, 'ε': 1104,
        'ζ': 1105, 'η': 1106, 'θ': 1107, 'ι': 1108, 'κ': 1109,
        'λ': 1110, 'μ': 1111, 'ν': 1112, 'ξ': 1113, 'ο': 1114,
        'π': 1115, 'ρ': 1116, 'σ': 1117, 'τ': 1118, 'υ': 1119,
        'φ': 1120, 'χ': 1121, 'ψ': 1122, 'ω': 1123,
        'Α': 1124, 'Β': 1125, 'Γ': 1126, 'Δ': 1127, 'Ε': 1128,
        'Ζ': 1129, 'Η': 1130, 'Θ': 1131, 'Ι': 1132, 'Κ': 1133,
        'Λ': 1134, 'Μ': 1135, 'Ν': 1136, 'Ξ': 1137, 'Ο': 1138,
        'Π': 1139, 'Ρ': 1140, 'Σ': 1141, 'Τ': 1142, 'Υ': 1143,
        'Φ': 1144, 'Χ': 1145, 'Ψ': 1146, 'Ω': 1147,
    }
    
    # Common LaTeX commands
    LATEX = {
        '\\frac': 1200, '\\sqrt': 1201, '\\sum': 1202, '\\prod': 1203,
        '\\int': 1204, '\\lim': 1205, '\\infty': 1206, '\\partial': 1207,
        '\\nabla': 1208, '\\forall': 1209, '\\exists': 1210, '\\in': 1211,
        '\\notin': 1212, '\\subset': 1213, '\\supset': 1214, '\\cup': 1215,
        '\\cap': 1216, '\\emptyset': 1217, '\\rightarrow': 1218, '\\leftarrow': 1219,
        '\\Rightarrow': 1220, '\\Leftarrow': 1221, '\\iff': 1222, '\\neg': 1223,
        '\\land': 1224, '\\lor': 1225, '\\oplus': 1226, '\\otimes': 1227,
        '\\cdot': 1228, '\\times': 1229, '\\div': 1230, '\\pm': 1231,
        '\\mp': 1232, '\\leq': 1233, '\\geq': 1234, '\\neq': 1235,
        '\\approx': 1236, '\\equiv': 1237, '\\cong': 1238, '\\sim': 1239,
        '\\propto': 1240, '\\perp': 1241, '\\parallel': 1242, '\\angle': 1243,
        '\\triangle': 1244, '\\square': 1245, '\\circ': 1246, '\\bullet': 1247,
        '\\star': 1248, '\\ast': 1249, '\\dagger': 1250, '\\ddagger': 1251,
        '\\prime': 1252, '\\ldots': 1253, '\\cdots': 1254, '\\vdots': 1255,
        '\\ddots': 1256, '\\binom': 1257, '\\choose': 1258, '\\pmod': 1259,
        '\\mod': 1260, '\\gcd': 1261, '\\lcm': 1262, '\\max': 1263,
        '\\min': 1264, '\\sup': 1265, '\\inf': 1266, '\\log': 1267,
        '\\ln': 1268, '\\exp': 1269, '\\sin': 1270, '\\cos': 1271,
        '\\tan': 1272, '\\cot': 1273, '\\sec': 1274, '\\csc': 1275,
        '\\arcsin': 1276, '\\arccos': 1277, '\\arctan': 1278, '\\sinh': 1279,
        '\\cosh': 1280, '\\tanh': 1281, '\\det': 1282, '\\dim': 1283,
        '\\ker': 1284, '\\hom': 1285, '\\arg': 1286, '\\deg': 1287,
        '\\left': 1288, '\\right': 1289, '\\big': 1290, '\\Big': 1291,
        '\\bigg': 1292, '\\Bigg': 1293, '\\text': 1294, '\\mathbf': 1295,
        '\\mathit': 1296, '\\mathrm': 1297, '\\mathcal': 1298, '\\mathbb': 1299,
        '\\boxed': 1300,
    }
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # Build full vocab
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Add special
        self.token_to_id.update(self.SPECIAL_TOKENS)
        
        # Add operators
        self.token_to_id.update(self.OPERATORS)
        
        # Add Greek
        self.token_to_id.update(self.GREEK)
        
        # Add LaTeX
        self.token_to_id.update(self.LATEX)
        
        # Add single letters as variables
        for i, c in enumerate('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            self.token_to_id[c] = self.WORD_START + i
        
        # Build reverse
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        # Word vocabulary (dynamically extended)
        self.word_vocab: Dict[str, int] = {}
        self.next_word_id = self.WORD_START + 100
    
    def get_id(self, token: str) -> int:
        """Get token ID, adding to vocab if new word."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        # Check word vocab
        if token in self.word_vocab:
            return self.word_vocab[token]
        
        # Add new word
        if self.next_word_id < self.vocab_size:
            self.word_vocab[token] = self.next_word_id
            self.id_to_token[self.next_word_id] = token
            self.next_word_id += 1
            return self.word_vocab[token]
        
        # Vocab full, return UNK
        return self.SPECIAL_TOKENS['<unk>']
    
    def get_token(self, token_id: int) -> str:
        """Get token string from ID."""
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
        return '<unk>'
    
    @property
    def pad_token_id(self) -> int:
        return self.SPECIAL_TOKENS['<pad>']
    
    @property
    def unk_token_id(self) -> int:
        return self.SPECIAL_TOKENS['<unk>']
    
    @property
    def bos_token_id(self) -> int:
        return self.SPECIAL_TOKENS['<bos>']
    
    @property
    def eos_token_id(self) -> int:
        return self.SPECIAL_TOKENS['<eos>']


# =============================================================================
# NUMBER ENCODER
# =============================================================================

class NumberEncoder:
    """
    Encodes numbers as token sequences.
    
    Strategy: Always use digit-by-digit encoding to avoid range collisions.
    """
    
    # Token IDs for digits (100-109)
    DIGIT_BASE = 100
    
    # Special number tokens
    DECIMAL_POINT = 110
    NEGATIVE = 111
    EXPONENT = 112  # For scientific notation
    
    # Range check
    NUMBER_TOKEN_MIN = 100
    NUMBER_TOKEN_MAX = 119
    
    @classmethod
    def is_number_token(cls, tid: int) -> bool:
        """Check if a token ID is a number token."""
        return cls.NUMBER_TOKEN_MIN <= tid <= cls.NUMBER_TOKEN_MAX
    
    @classmethod
    def encode_integer(cls, n: int) -> List[int]:
        """Encode an integer digit by digit."""
        tokens = []
        
        # Handle negative
        if n < 0:
            tokens.append(cls.NEGATIVE)
            n = -n
        
        # Handle zero
        if n == 0:
            return tokens + [cls.DIGIT_BASE]
        
        # Encode digits
        digits = str(n)
        for d in digits:
            tokens.append(cls.DIGIT_BASE + int(d))
        
        return tokens
    
    @classmethod
    def encode_float(cls, f: float) -> List[int]:
        """Encode a float."""
        tokens = []
        
        # Handle negative
        if f < 0:
            tokens.append(cls.NEGATIVE)
            f = -f
        
        # Split into integer and fractional parts
        int_part = int(f)
        frac_part = f - int_part
        
        # Encode integer part
        if int_part == 0:
            tokens.append(cls.DIGIT_BASE)  # Leading zero
        else:
            for d in str(int_part):
                tokens.append(cls.DIGIT_BASE + int(d))
        
        # Encode fractional part
        if frac_part > 1e-10:
            tokens.append(cls.DECIMAL_POINT)
            
            # Get fractional digits (up to 6)
            frac_str = f'{frac_part:.6f}'[2:].rstrip('0')
            for d in frac_str[:6]:
                tokens.append(cls.DIGIT_BASE + int(d))
        
        return tokens
    
    @classmethod
    def encode_scientific(cls, s: str) -> List[int]:
        """Encode scientific notation."""
        tokens = []
        
        # Split on e/E
        match = re.match(r'(-?\d+\.?\d*)([eE])([+-]?\d+)', s)
        if match:
            mantissa = float(match.group(1))
            exp = int(match.group(3))
            
            tokens.extend(cls.encode_float(mantissa))
            tokens.append(cls.EXPONENT)
            tokens.extend(cls.encode_integer(exp))
        
        return tokens
    
    @classmethod
    def decode(cls, tokens: List[int]) -> Optional[float]:
        """Decode tokens back to number."""
        if not tokens:
            return None
        
        negative = False
        integer_part = 0
        fractional_part = 0.0
        frac_divisor = 1.0
        in_fraction = False
        exponent = 0
        in_exponent = False
        exp_negative = False
        have_digit = False
        
        for t in tokens:
            if t == cls.NEGATIVE:
                if in_exponent:
                    exp_negative = True
                else:
                    negative = True
            elif t == cls.DECIMAL_POINT:
                in_fraction = True
            elif t == cls.EXPONENT:
                in_exponent = True
            elif cls.DIGIT_BASE <= t < cls.DIGIT_BASE + 10:
                # Digit 0-9
                digit = t - cls.DIGIT_BASE
                if in_exponent:
                    exponent = exponent * 10 + digit
                elif in_fraction:
                    frac_divisor *= 10
                    fractional_part += digit / frac_divisor
                else:
                    integer_part = integer_part * 10 + digit
                have_digit = True
        
        if not have_digit:
            return None
        
        result = integer_part + fractional_part
        if negative:
            result = -result
        
        if in_exponent:
            if exp_negative:
                exponent = -exponent
            result *= (10 ** exponent)
        
        return result


# =============================================================================
# MAIN TOKENIZER
# =============================================================================

class MathTokenizer:
    """
    Main math tokenizer.
    
    Drop-in replacement for HuggingFace tokenizers.
    
    Usage:
        tokenizer = MathTokenizer()
        tokens = tokenizer.encode("Let x = 42. Then x^2 = 1764.")
        text = tokenizer.decode(tokens)
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.vocab = MathVocab(vocab_size)
        self.lexer = MathLexer()
        self.number_encoder = NumberEncoder()
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Cache for common tokens
        self._cache: Dict[str, List[int]] = {}
        self._cache_size = 10000
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Any:
        """
        Encode text to token IDs.
        
        Compatible with HuggingFace API.
        """
        # Check cache
        cache_key = text[:100]  # Use prefix as key
        if cache_key in self._cache and len(text) <= 100:
            token_ids = self._cache[cache_key].copy()
        else:
            token_ids = self._encode_impl(text)
            
            # Cache short sequences
            if len(text) <= 100 and len(self._cache) < self._cache_size:
                self._cache[cache_key] = token_ids.copy()
        
        # Add special tokens
        if add_special_tokens:
            if self.add_bos:
                token_ids = [self.vocab.bos_token_id] + token_ids
            if self.add_eos:
                token_ids = token_ids + [self.vocab.eos_token_id]
        
        # Truncation
        if truncation and max_length:
            token_ids = token_ids[:max_length]
        
        # Padding
        if padding and max_length:
            while len(token_ids) < max_length:
                token_ids.append(self.vocab.pad_token_id)
        
        # Return format
        if return_tensors == 'pt':
            import torch
            return torch.tensor([token_ids])
        
        return token_ids
    
    def _encode_impl(self, text: str) -> List[int]:
        """Internal encoding implementation."""
        token_ids = []
        
        for token in self.lexer.tokenize(text):
            if token.token_type == TokenType.WHITESPACE:
                # Skip whitespace (or use special token)
                continue
            
            elif token.token_type == TokenType.NEWLINE:
                # Could use special newline token
                continue
            
            elif token.token_type in (TokenType.INTEGER, TokenType.FLOAT):
                # Encode number
                if token.numeric_value is not None:
                    if token.token_type == TokenType.INTEGER:
                        token_ids.extend(
                            self.number_encoder.encode_integer(int(token.numeric_value))
                        )
                    else:
                        token_ids.extend(
                            self.number_encoder.encode_float(token.numeric_value)
                        )
                else:
                    token_ids.append(self.vocab.get_id(token.text))
            
            elif token.token_type == TokenType.SCIENTIFIC:
                token_ids.extend(
                    self.number_encoder.encode_scientific(token.text)
                )
            
            elif token.token_type == TokenType.FRACTION:
                # Encode as num/num
                parts = token.text.split('/')
                token_ids.extend(self.number_encoder.encode_integer(int(parts[0])))
                token_ids.append(self.vocab.get_id('/'))
                token_ids.extend(self.number_encoder.encode_integer(int(parts[1])))
            
            elif token.token_type == TokenType.GREEK:
                token_ids.append(self.vocab.get_id(token.text))
            
            elif token.token_type in (TokenType.LATEX_CMD, TokenType.LATEX_FRAC,
                                       TokenType.LATEX_SQRT, TokenType.LATEX_SUM,
                                       TokenType.LATEX_PROD, TokenType.LATEX_INT,
                                       TokenType.LATEX_LIM):
                token_ids.append(self.vocab.get_id(token.text))
            
            elif token.token_type in (TokenType.PLUS, TokenType.MINUS, TokenType.TIMES,
                                       TokenType.DIVIDE, TokenType.EQUALS, TokenType.LESS,
                                       TokenType.GREATER, TokenType.LEQ, TokenType.GEQ,
                                       TokenType.NEQ, TokenType.APPROX):
                token_ids.append(self.vocab.get_id(token.text))
            
            elif token.token_type in (TokenType.LPAREN, TokenType.RPAREN,
                                       TokenType.LBRACKET, TokenType.RBRACKET,
                                       TokenType.LBRACE, TokenType.RBRACE):
                token_ids.append(self.vocab.get_id(token.text))
            
            elif token.token_type == TokenType.VARIABLE:
                token_ids.append(self.vocab.get_id(token.text))
            
            elif token.token_type == TokenType.WORD:
                # Check for proof keywords
                lower = token.text.lower()
                if lower in ('qed', 'proof', 'therefore', 'hence', 'thus'):
                    token_ids.append(self.vocab.get_id(f'<{lower}>') 
                                    if f'<{lower}>' in self.vocab.token_to_id 
                                    else self.vocab.get_id(token.text))
                else:
                    token_ids.append(self.vocab.get_id(token.text))
            
            elif token.token_type == TokenType.PUNCTUATION:
                token_ids.append(self.vocab.get_id(token.text))
            
            else:
                # Unknown
                token_ids.append(self.vocab.unk_token_id)
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        parts = []
        i = 0
        
        while i < len(token_ids):
            tid = token_ids[i]
            
            # Skip special tokens
            if skip_special_tokens and tid in (
                self.vocab.pad_token_id,
                self.vocab.bos_token_id,
                self.vocab.eos_token_id,
            ):
                i += 1
                continue
            
            # Check if it's a number sequence
            if self._is_number_token(tid):
                # Collect number tokens
                num_tokens = []
                while i < len(token_ids) and self._is_number_token(token_ids[i]):
                    num_tokens.append(token_ids[i])
                    i += 1
                
                # Decode number
                value = NumberEncoder.decode(num_tokens)
                if value is not None:
                    if value == int(value):
                        parts.append(str(int(value)))
                    else:
                        parts.append(f'{value:.6g}')
                continue
            
            # Regular token
            token_str = self.vocab.get_token(tid)
            if token_str != '<unk>':
                parts.append(token_str)
            
            i += 1
        
        # Join with appropriate spacing
        result = ''
        for i, part in enumerate(parts):
            if i > 0 and part not in '.,;:!?)]}' and (not result or result[-1] not in '([{'):
                result += ' '
            result += part
        
        return result
    
    def _is_number_token(self, tid: int) -> bool:
        """Check if token ID is part of a number."""
        return NumberEncoder.is_number_token(tid)
    
    def __call__(
        self,
        text: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """HuggingFace-style callable interface."""
        token_ids = self.encode(text, **kwargs)
        
        if isinstance(token_ids, list):
            attention_mask = [1] * len(token_ids)
            return {
                'input_ids': token_ids,
                'attention_mask': attention_mask,
            }
        else:
            # Tensor
            import torch
            attention_mask = torch.ones_like(token_ids)
            return {
                'input_ids': token_ids,
                'attention_mask': attention_mask,
            }
    
    # Properties for HuggingFace compatibility
    @property
    def pad_token_id(self) -> int:
        return self.vocab.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.vocab.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self.vocab.bos_token_id
    
    @property
    def unk_token_id(self) -> int:
        return self.vocab.unk_token_id
    
    def save_pretrained(self, path: str):
        """Save tokenizer to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vocab
        vocab_data = {
            'token_to_id': self.vocab.token_to_id,
            'word_vocab': self.vocab.word_vocab,
            'next_word_id': self.vocab.next_word_id,
            'vocab_size': self.vocab.vocab_size,
        }
        
        with open(path / 'vocab.json', 'w') as f:
            json.dump(vocab_data, f)
        
        # Save config
        config = {
            'add_bos': self.add_bos,
            'add_eos': self.add_eos,
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'MathTokenizer':
        """Load tokenizer from directory."""
        path = Path(path)
        
        with open(path / 'config.json') as f:
            config = json.load(f)
        
        with open(path / 'vocab.json') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(
            vocab_size=vocab_data['vocab_size'],
            add_bos=config['add_bos'],
            add_eos=config['add_eos'],
        )
        
        # Restore vocab
        tokenizer.vocab.token_to_id.update(vocab_data['token_to_id'])
        tokenizer.vocab.word_vocab = vocab_data['word_vocab']
        tokenizer.vocab.next_word_id = vocab_data['next_word_id']
        tokenizer.vocab.id_to_token = {v: k for k, v in tokenizer.vocab.token_to_id.items()}
        
        return tokenizer


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'MathTokenizer',
    'MathLexer',
    'MathVocab',
    'NumberEncoder',
    'Token',
    'TokenType',
]


if __name__ == "__main__":
    print("MathTokenizer 2.0")
    print("=================")
    print()
    
    tokenizer = MathTokenizer()
    
    # Test cases
    tests = [
        "Let x = 42.",
        "Prove that 2 + 2 = 4.",
        "∑_{i=1}^{n} i = n(n+1)/2",
        "The answer is \\boxed{1764}.",
        "If x > 0 and y < 10, then x + y ≤ 10.",
        "π ≈ 3.14159",
        "2.5e-10 is a small number.",
    ]
    
    for text in tests:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        print(f"Input:   {text}")
        print(f"Tokens:  {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"Decoded: {decoded}")
        print()
