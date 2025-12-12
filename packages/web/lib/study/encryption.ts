/**
 * Study Book Encryption System
 *
 * Provides encryption, salting, and integrity verification for chat history.
 * Uses AES-256-GCM for encryption and SHA-256 for integrity hashing.
 *
 * Security features:
 * - AES-256-GCM authenticated encryption
 * - Per-conversation salt (12 bytes)
 * - Integrity hash (SHA-256) to detect tampering
 * - Automatic key derivation from user + conversation context
 */

// Encryption uses Web Crypto API (available in Node.js and browsers)
const ALGORITHM = 'AES-GCM';
const KEY_LENGTH = 256;
const IV_LENGTH = 12; // 96 bits for GCM
const SALT_LENGTH = 16;
const TAG_LENGTH = 128; // GCM auth tag bits

// =============================================================================
// KEY DERIVATION
// =============================================================================

/**
 * Derive an encryption key from password + salt using PBKDF2
 */
async function deriveKey(
  password: string,
  salt: Uint8Array
): Promise<CryptoKey> {
  const encoder = new TextEncoder();
  const passwordBuffer = encoder.encode(password);

  // Import password as raw key material
  const baseKey = await crypto.subtle.importKey(
    'raw',
    passwordBuffer,
    'PBKDF2',
    false,
    ['deriveBits', 'deriveKey']
  );

  // Derive AES key using PBKDF2
  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    baseKey,
    { name: ALGORITHM, length: KEY_LENGTH },
    false,
    ['encrypt', 'decrypt']
  );
}

/**
 * Generate a secure random salt
 */
function generateSalt(): Uint8Array {
  return crypto.getRandomValues(new Uint8Array(SALT_LENGTH));
}

/**
 * Generate a secure random IV
 */
function generateIV(): Uint8Array {
  return crypto.getRandomValues(new Uint8Array(IV_LENGTH));
}

// =============================================================================
// ENCRYPTION / DECRYPTION
// =============================================================================

export interface EncryptedPayload {
  /** Base64-encoded ciphertext */
  ciphertext: string;
  /** Base64-encoded IV */
  iv: string;
  /** Base64-encoded salt (for key derivation) */
  salt: string;
  /** SHA-256 hash of original plaintext for integrity verification */
  integrityHash: string;
  /** Timestamp when encrypted */
  encryptedAt: string;
  /** Version for future compatibility */
  version: number;
}

/**
 * Encrypt data with AES-256-GCM
 */
export async function encrypt(
  plaintext: string,
  password: string
): Promise<EncryptedPayload> {
  const encoder = new TextEncoder();
  const plaintextBuffer = encoder.encode(plaintext);

  // Generate salt and IV
  const salt = generateSalt();
  const iv = generateIV();

  // Derive key
  const key = await deriveKey(password, salt);

  // Encrypt
  const ciphertextBuffer = await crypto.subtle.encrypt(
    { name: ALGORITHM, iv, tagLength: TAG_LENGTH },
    key,
    plaintextBuffer
  );

  // Calculate integrity hash of plaintext
  const hashBuffer = await crypto.subtle.digest('SHA-256', plaintextBuffer);
  const integrityHash = bufferToBase64(new Uint8Array(hashBuffer));

  return {
    ciphertext: bufferToBase64(new Uint8Array(ciphertextBuffer)),
    iv: bufferToBase64(iv),
    salt: bufferToBase64(salt),
    integrityHash,
    encryptedAt: new Date().toISOString(),
    version: 1,
  };
}

/**
 * Decrypt data and verify integrity
 */
export async function decrypt(
  payload: EncryptedPayload,
  password: string
): Promise<{ plaintext: string; verified: boolean }> {
  const decoder = new TextDecoder();

  // Decode from base64
  const ciphertext = base64ToBuffer(payload.ciphertext);
  const iv = base64ToBuffer(payload.iv);
  const salt = base64ToBuffer(payload.salt);

  // Derive key
  const key = await deriveKey(password, salt);

  // Decrypt
  const plaintextBuffer = await crypto.subtle.decrypt(
    { name: ALGORITHM, iv, tagLength: TAG_LENGTH },
    key,
    ciphertext
  );

  const plaintext = decoder.decode(plaintextBuffer);

  // Verify integrity
  const encoder = new TextEncoder();
  const verifyBuffer = await crypto.subtle.digest(
    'SHA-256',
    encoder.encode(plaintext)
  );
  const verifyHash = bufferToBase64(new Uint8Array(verifyBuffer));
  const verified = verifyHash === payload.integrityHash;

  return { plaintext, verified };
}

// =============================================================================
// CONVERSATION ENCRYPTION
// =============================================================================

export interface EncryptedConversation {
  conversationId: string;
  userId: string;
  payload: EncryptedPayload;
  metadata: {
    mode: string;
    messageCount: number;
    title?: string;
    createdAt: string;
    lastMessageAt: string;
  };
}

/**
 * Encrypt an entire conversation
 */
export async function encryptConversation(
  conversation: {
    id: string;
    userId: string;
    mode: string;
    title?: string;
    messages: Array<{
      role: string;
      content: string;
      timestamp: string;
      metadata?: Record<string, unknown>;
    }>;
    createdAt: string;
    lastMessageAt: string;
  },
  encryptionKey: string
): Promise<EncryptedConversation> {
  // Build password from user context + conversation ID + key
  const password = `${conversation.userId}:${conversation.id}:${encryptionKey}`;

  // Serialize messages
  const plaintext = JSON.stringify(conversation.messages);

  // Encrypt
  const payload = await encrypt(plaintext, password);

  return {
    conversationId: conversation.id,
    userId: conversation.userId,
    payload,
    metadata: {
      mode: conversation.mode,
      messageCount: conversation.messages.length,
      title: conversation.title,
      createdAt: conversation.createdAt,
      lastMessageAt: conversation.lastMessageAt,
    },
  };
}

/**
 * Decrypt a conversation and verify integrity
 */
export async function decryptConversation(
  encrypted: EncryptedConversation,
  encryptionKey: string
): Promise<{
  messages: Array<{
    role: string;
    content: string;
    timestamp: string;
    metadata?: Record<string, unknown>;
  }>;
  verified: boolean;
  tampered: boolean;
}> {
  // Build password
  const password = `${encrypted.userId}:${encrypted.conversationId}:${encryptionKey}`;

  try {
    const { plaintext, verified } = await decrypt(encrypted.payload, password);
    const messages = JSON.parse(plaintext);

    return {
      messages,
      verified,
      tampered: !verified,
    };
  } catch (error) {
    // Decryption failed - likely tampered or wrong key
    return {
      messages: [],
      verified: false,
      tampered: true,
    };
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function bufferToBase64(buffer: Uint8Array): string {
  // Browser-compatible base64 encoding
  const binary = String.fromCharCode(...buffer);
  return btoa(binary);
}

function base64ToBuffer(base64: string): Uint8Array {
  // Browser-compatible base64 decoding
  const binary = atob(base64);
  const buffer = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    buffer[i] = binary.charCodeAt(i);
  }
  return buffer;
}

/**
 * Generate a secure encryption key for a user
 * This should be stored securely (not in the database with the encrypted data)
 */
export function generateEncryptionKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return bufferToBase64(bytes);
}

/**
 * Hash a conversation ID with user context for additional security
 */
export async function hashConversationId(
  conversationId: string,
  userId: string,
  salt: string
): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(`${userId}:${conversationId}:${salt}`);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  return bufferToBase64(new Uint8Array(hashBuffer));
}
