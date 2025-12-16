/**
 * Simple encryption utilities for secure data handling
 * Uses Web Crypto API for browser-compatible encryption
 */

export interface EncryptedPayload {
  ciphertext: string;
  iv: string;
  salt: string;
  integrityHash: string;
  encryptedAt: string;
  version: number;
}

export interface DecryptedPayload {
  plaintext: string;
  verified: boolean;
}

/**
 * Encrypt data using AES-GCM with PBKDF2 key derivation
 */
export async function encrypt(plaintext: string, key: string): Promise<EncryptedPayload> {
  const encoder = new TextEncoder();
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));

  // Derive key using PBKDF2
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(key),
    'PBKDF2',
    false,
    ['deriveKey']
  );

  const derivedKey = await crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt']
  );

  // Encrypt
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    derivedKey,
    encoder.encode(plaintext)
  );

  return {
    ciphertext: btoa(String.fromCharCode(...new Uint8Array(encrypted))),
    iv: btoa(String.fromCharCode(...iv)),
    salt: btoa(String.fromCharCode(...salt)),
    integrityHash: await sha256(plaintext),
    encryptedAt: new Date().toISOString(),
    version: 1,
  };
}

/**
 * Decrypt data using AES-GCM with PBKDF2 key derivation
 */
export async function decrypt(
  payload: EncryptedPayload,
  key: string
): Promise<DecryptedPayload> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  // Decode base64
  const ciphertext = Uint8Array.from(atob(payload.ciphertext), c => c.charCodeAt(0));
  const iv = Uint8Array.from(atob(payload.iv), c => c.charCodeAt(0));
  const salt = Uint8Array.from(atob(payload.salt), c => c.charCodeAt(0));

  // Derive key using PBKDF2
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(key),
    'PBKDF2',
    false,
    ['deriveKey']
  );

  const derivedKey = await crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['decrypt']
  );

  // Decrypt
  const decrypted = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv },
    derivedKey,
    ciphertext
  );

  const plaintext = decoder.decode(decrypted);

  // Verify integrity if hash was provided
  let verified = true;
  if (payload.integrityHash) {
    const currentHash = await sha256(plaintext);
    verified = currentHash === payload.integrityHash;
  }

  return { plaintext, verified };
}

/**
 * SHA-256 hash of a string
 */
export async function sha256(message: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(message);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Derive an encryption key from user ID
 */
export function deriveKey(userId: string): string {
  // In production, this should use a proper KDF with a secret salt
  return `lf-${userId}-${process.env.ANONYMIZATION_SALT || 'default-salt'}`;
}
