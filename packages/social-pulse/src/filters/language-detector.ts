/**
 * Language Detection Filter
 *
 * Detects language from text content using character patterns,
 * common words, and script identification.
 *
 * Lightweight implementation - no external dependencies.
 */

import type { PostFilter, SocialPost } from '../types.js';

/**
 * Language detection configuration
 */
export interface LanguageDetectorConfig {
  /** Languages to filter for (empty = allow all) */
  allowedLanguages?: string[];
  /** Minimum confidence to accept detection */
  minConfidence?: number;
}

/**
 * Language profile with common words and character patterns
 */
interface LanguageProfile {
  code: string;
  name: string;
  /** Common words (stopwords) */
  commonWords: Set<string>;
  /** Character ranges (for non-Latin scripts) */
  scriptRanges?: Array<[number, number]>;
}

// Language profiles for detection
const LANGUAGE_PROFILES: LanguageProfile[] = [
  {
    code: 'en',
    name: 'English',
    commonWords: new Set([
      'the',
      'be',
      'to',
      'of',
      'and',
      'a',
      'in',
      'that',
      'have',
      'i',
      'it',
      'for',
      'not',
      'on',
      'with',
      'he',
      'as',
      'you',
      'do',
      'at',
      'this',
      'but',
      'his',
      'by',
      'from',
      'they',
      'we',
      'say',
      'her',
      'she',
      'or',
      'an',
      'will',
      'my',
      'one',
      'all',
      'would',
      'there',
      'their',
      'what',
      'is',
      'are',
      'was',
      'were',
      'been',
      'being',
      'has',
      'had',
      'having',
    ]),
  },
  {
    code: 'es',
    name: 'Spanish',
    commonWords: new Set([
      'de',
      'la',
      'que',
      'el',
      'en',
      'y',
      'a',
      'los',
      'del',
      'se',
      'las',
      'por',
      'un',
      'para',
      'con',
      'no',
      'una',
      'su',
      'al',
      'es',
      'lo',
      'como',
      'más',
      'pero',
      'sus',
      'le',
      'ya',
      'o',
      'este',
      'si',
      'porque',
      'esta',
      'entre',
      'cuando',
      'muy',
      'sin',
      'sobre',
      'ser',
      'tiene',
      'también',
    ]),
  },
  {
    code: 'fr',
    name: 'French',
    commonWords: new Set([
      'le',
      'de',
      'un',
      'être',
      'et',
      'à',
      'il',
      'avoir',
      'ne',
      'je',
      'son',
      'que',
      'se',
      'qui',
      'ce',
      'dans',
      'en',
      'du',
      'elle',
      'au',
      'pour',
      'pas',
      'que',
      'vous',
      'par',
      'sur',
      'faire',
      'plus',
      'dire',
      'me',
      'on',
      'mon',
      'lui',
      'nous',
      'comme',
      'mais',
      'pouvoir',
      'avec',
      'tout',
      'y',
    ]),
  },
  {
    code: 'de',
    name: 'German',
    commonWords: new Set([
      'der',
      'die',
      'und',
      'in',
      'den',
      'von',
      'zu',
      'das',
      'mit',
      'sich',
      'des',
      'auf',
      'für',
      'ist',
      'im',
      'dem',
      'nicht',
      'ein',
      'eine',
      'als',
      'auch',
      'es',
      'an',
      'er',
      'hat',
      'aus',
      'bei',
      'sind',
      'noch',
      'nach',
      'wird',
      'einer',
      'um',
      'am',
      'haben',
      'war',
      'wie',
      'oder',
      'werden',
      'dass',
    ]),
  },
  {
    code: 'pt',
    name: 'Portuguese',
    commonWords: new Set([
      'de',
      'a',
      'o',
      'que',
      'e',
      'do',
      'da',
      'em',
      'um',
      'para',
      'é',
      'com',
      'não',
      'uma',
      'os',
      'no',
      'se',
      'na',
      'por',
      'mais',
      'as',
      'dos',
      'como',
      'mas',
      'foi',
      'ao',
      'ele',
      'das',
      'tem',
      'à',
      'seu',
      'sua',
      'ou',
      'ser',
      'quando',
      'muito',
      'há',
      'nos',
      'já',
      'está',
    ]),
  },
  {
    code: 'ru',
    name: 'Russian',
    commonWords: new Set([
      'и',
      'в',
      'не',
      'на',
      'я',
      'что',
      'он',
      'с',
      'как',
      'это',
      'все',
      'она',
      'так',
      'его',
      'но',
      'да',
      'ты',
      'к',
      'у',
      'же',
      'вы',
      'за',
      'бы',
      'по',
      'только',
      'её',
      'мне',
      'было',
      'вот',
      'от',
    ]),
    scriptRanges: [[0x0400, 0x04ff]], // Cyrillic
  },
  {
    code: 'ar',
    name: 'Arabic',
    commonWords: new Set([
      'في',
      'من',
      'على',
      'إلى',
      'أن',
      'هذا',
      'الذي',
      'التي',
      'هو',
      'كان',
      'لم',
      'عن',
      'مع',
      'هي',
      'كل',
      'لا',
      'ما',
      'قد',
      'بين',
      'إن',
    ]),
    scriptRanges: [[0x0600, 0x06ff]], // Arabic
  },
  {
    code: 'zh',
    name: 'Chinese',
    commonWords: new Set([
      '的',
      '一',
      '是',
      '不',
      '了',
      '在',
      '人',
      '有',
      '我',
      '他',
      '这',
      '中',
      '大',
      '来',
      '上',
      '国',
      '个',
      '到',
      '说',
      '们',
      '为',
      '子',
      '和',
      '你',
      '地',
      '出',
      '道',
      '也',
      '时',
      '年',
    ]),
    scriptRanges: [[0x4e00, 0x9fff]], // CJK Unified Ideographs
  },
  {
    code: 'ja',
    name: 'Japanese',
    commonWords: new Set([
      'の',
      'に',
      'は',
      'を',
      'た',
      'が',
      'で',
      'て',
      'と',
      'し',
      'れ',
      'さ',
      'ある',
      'いる',
      'も',
      'する',
      'から',
      'な',
      'こと',
      'として',
    ]),
    scriptRanges: [
      [0x3040, 0x309f], // Hiragana
      [0x30a0, 0x30ff], // Katakana
    ],
  },
  {
    code: 'ko',
    name: 'Korean',
    commonWords: new Set([
      '이',
      '그',
      '저',
      '것',
      '수',
      '나',
      '등',
      '및',
      '년',
      '에',
      '의',
      '을',
      '를',
      '가',
      '은',
      '는',
      '와',
      '과',
      '도',
      '에서',
    ]),
    scriptRanges: [[0xac00, 0xd7af]], // Hangul Syllables
  },
  {
    code: 'hi',
    name: 'Hindi',
    commonWords: new Set([
      'का',
      'के',
      'को',
      'में',
      'है',
      'और',
      'से',
      'की',
      'हैं',
      'पर',
      'इस',
      'एक',
      'होता',
      'भी',
      'ने',
      'तो',
      'या',
      'था',
      'करने',
      'हो',
    ]),
    scriptRanges: [[0x0900, 0x097f]], // Devanagari
  },
  {
    code: 'fa',
    name: 'Persian',
    commonWords: new Set([
      'و',
      'در',
      'به',
      'از',
      'که',
      'این',
      'را',
      'با',
      'است',
      'آن',
      'برای',
      'یک',
      'هم',
      'تا',
      'بود',
      'خود',
      'ها',
      'می',
      'او',
      'شد',
    ]),
    scriptRanges: [[0x0600, 0x06ff]], // Arabic (shared)
  },
  {
    code: 'tr',
    name: 'Turkish',
    commonWords: new Set([
      'bir',
      've',
      'bu',
      'için',
      'de',
      'da',
      'ile',
      'gibi',
      'o',
      'en',
      'çok',
      'ne',
      'daha',
      'var',
      'olan',
      'kadar',
      'sonra',
      'olarak',
      'ancak',
      'şey',
      'ise',
      'yok',
      'göre',
      'olan',
      'oldu',
      'olur',
      'olmuş',
      'olmak',
      'değil',
      'mı',
    ]),
  },
  {
    code: 'uk',
    name: 'Ukrainian',
    commonWords: new Set([
      'і',
      'в',
      'не',
      'на',
      'що',
      'з',
      'як',
      'це',
      'він',
      'та',
      'до',
      'є',
      'але',
      'за',
      'від',
      'його',
      'по',
      'вона',
      'про',
      'так',
    ]),
    scriptRanges: [[0x0400, 0x04ff]], // Cyrillic
  },
];

export class LanguageDetector implements PostFilter {
  readonly name = 'language-detector';
  private config: Required<LanguageDetectorConfig>;
  private allowedSet: Set<string>;

  constructor(config: LanguageDetectorConfig = {}) {
    this.config = {
      allowedLanguages: config.allowedLanguages ?? [],
      minConfidence: config.minConfidence ?? 0.3,
    };
    this.allowedSet = new Set(this.config.allowedLanguages);
  }

  /**
   * Process a post and attach detected language
   */
  async process(post: SocialPost): Promise<SocialPost | null> {
    const { language, confidence } = this.detect(post.content);

    // Attach detected language
    const processedPost: SocialPost = {
      ...post,
    };

    // Only set language if it has a value
    const newLanguage = confidence >= this.config.minConfidence ? language : post.language;
    if (newLanguage) {
      processedPost.language = newLanguage;
    }

    // Filter if language restrictions are set
    if (this.allowedSet.size > 0 && !this.allowedSet.has(language)) {
      return null;
    }

    return processedPost;
  }

  /**
   * Detect language of text content
   */
  detect(text: string): {
    language: string;
    confidence: number;
    alternatives: Array<{ code: string; score: number }>;
  } {
    if (!text || text.trim().length < 3) {
      return { language: 'unknown', confidence: 0, alternatives: [] };
    }

    // First, check for non-Latin scripts (faster and more accurate)
    const scriptResult = this.detectByScript(text);
    if (scriptResult.confidence > 0.8) {
      return { ...scriptResult, alternatives: [] };
    }

    // Then, use word-based detection
    const scores = new Map<string, number>();
    const words = text.toLowerCase().match(/[\p{L}]+/gu) ?? [];

    if (words.length === 0) {
      return { language: 'unknown', confidence: 0, alternatives: [] };
    }

    for (const profile of LANGUAGE_PROFILES) {
      let score = 0;
      let matches = 0;

      for (const word of words) {
        if (profile.commonWords.has(word)) {
          matches++;
        }
      }

      // Calculate score based on match ratio
      score = matches / words.length;

      // Boost if script matches
      if (profile.scriptRanges && scriptResult.language === profile.code) {
        score *= 1.5;
      }

      scores.set(profile.code, score);
    }

    // Sort by score
    const sorted = [...scores.entries()].sort((a, b) => b[1] - a[1]);

    if (sorted.length === 0 || sorted[0]![1] === 0) {
      // Fall back to script detection
      if (scriptResult.confidence > 0.3) {
        return { ...scriptResult, alternatives: [] };
      }
      return { language: 'unknown', confidence: 0, alternatives: [] };
    }

    const topScore = sorted[0]![1];
    const secondScore = sorted[1]?.[1] ?? 0;

    // Calculate confidence based on score gap
    let confidence = topScore;
    if (secondScore > 0) {
      // Higher gap = higher confidence
      confidence *= 1 + (topScore - secondScore);
    }
    confidence = Math.min(1, confidence);

    const alternatives = sorted.slice(1, 4).map(([code, score]) => ({ code, score }));

    return {
      language: sorted[0]![0],
      confidence,
      alternatives,
    };
  }

  /**
   * Detect language by character script
   */
  private detectByScript(text: string): { language: string; confidence: number } {
    const charCounts = new Map<string, number>();

    for (const char of text) {
      const code = char.charCodeAt(0);

      for (const profile of LANGUAGE_PROFILES) {
        if (!profile.scriptRanges) continue;

        for (const [start, end] of profile.scriptRanges) {
          if (code >= start && code <= end) {
            charCounts.set(profile.code, (charCounts.get(profile.code) ?? 0) + 1);
            break;
          }
        }
      }
    }

    // Find dominant script
    let maxCount = 0;
    let maxLang = 'unknown';
    let total = 0;

    for (const [lang, count] of charCounts) {
      total += count;
      if (count > maxCount) {
        maxCount = count;
        maxLang = lang;
      }
    }

    // Calculate confidence based on script proportion
    const confidence = total > 0 ? maxCount / text.length : 0;

    return { language: maxLang, confidence };
  }

  /**
   * Get all supported language codes
   */
  getSupportedLanguages(): string[] {
    return LANGUAGE_PROFILES.map((p) => p.code);
  }
}
