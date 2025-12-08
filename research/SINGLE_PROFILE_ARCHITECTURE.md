# SINGLE PROFILE ARCHITECTURE

**One Device = One Profile = One User**

## Rationale

### Security Benefits

1. **Prevents Multi-Accounting**
   - Can't create fresh accounts to farm achievements
   - Can't share subscription across multiple "profiles"
   - Makes ban evasion harder (need new device)

2. **Simplifies Data Model**
   - No profile switching logic
   - No "which profile is premium?" confusion
   - One user ID = one device ID = one Firebase UID

3. **Better Analytics**
   - True user count (not inflated by multiple profiles)
   - Accurate engagement metrics
   - Clearer subscription conversion funnel

4. **Prevents Abuse**
   - Can't create "test" profile to try exploits
   - Can't separate "clean" vs "cheating" profiles
   - Forces commitment to one identity

## Changes Required

### Remove Profile Switching

**Delete**:
- ✅ `ProfileSelectScreen.js` (no longer needed)
- ✅ Multi-profile logic in WelcomeScreen
- ✅ "Switch Profile" buttons
- ✅ Profile import/export (replaced by cloud sync)

**Simplify**:
- Profile setup happens ONCE on first launch
- User creates PIN/password during onboarding
- Can edit profile details, but can't switch/delete

### Data Model

**Before** (Multi-Profile):
```javascript
AsyncStorage:
  @veilpath_user_profiles: [profile1, profile2, profile3]
  @veilpath_active_profile: "profile2"
  @veilpath_readings: [...] // shared across profiles?
```

**After** (Single Profile):
```javascript
AsyncStorage:
  @veilpath_user_profile: {single profile object}
  @veilpath_readings: [...] // belongs to THE user
  @veilpath_user_pin: "hashed_pin" // REQUIRED

Firebase:
  /users/{deviceUID}/
    profile: {single profile}
    data/
      readings: [...]
      journal: [...]
      oracle_history: [...]
```

### PIN/Password Requirement

**On First Launch**:
1. Welcome screen
2. Profile setup (name, birthday, MBTI)
3. **CREATE PIN** (4-6 digit numeric or alphanumeric password)
4. Confirm PIN
5. Store hashed PIN locally
6. Main app

**On Subsequent Launches**:
1. Show lock screen
2. Enter PIN
3. Verify against stored hash
4. If correct: unlock app
5. If incorrect: lock out after 5 attempts (optional: biometric unlock)

**Security**:
- PIN hashed with bcrypt/scrypt before storage
- Optional: Biometric unlock (Face ID, Touch ID, fingerprint)
- Optional: 5 failed attempts = 5 minute lockout
- Optional: 10 failed attempts = factory reset warning

## Implementation Plan

### Phase 1: Remove Multi-Profile UI

```javascript
// Remove from App.js
- ProfileSelectScreen
- Profile switching navigation

// Simplify WelcomeScreen
- No "Select Profile" button
- Just "Get Started" (if new user)
- Or unlock screen (if returning user)

// Remove from SettingsScreen
- "Switch Profile" option
- "Export/Import Profile" (replaced by cloud sync)
```

### Phase 2: Add PIN Lock Screen

```javascript
// New screen: PINLockScreen.js
- Shows on app launch if PIN exists
- Numeric keypad (4-6 digits)
- Biometric option (optional)
- "Forgot PIN?" → security questions or email reset

// New screen: PINSetupScreen.js
- Shows during onboarding
- Create PIN (4-6 digits)
- Confirm PIN
- Explain: "Your PIN protects your personal readings and journal"
```

### Phase 3: Update Data Flow

```javascript
// On app launch:
const hasProfile = await AsyncStorage.getItem('@veilpath_user_profile');
const hasPIN = await AsyncStorage.getItem('@veilpath_user_pin');

if (!hasProfile) {
  // New user flow
  navigation.navigate('Onboarding'); // → ProfileSetup → PINSetup → Welcome
} else if (!hasPIN) {
  // Legacy user (had profile before PIN requirement)
  navigation.navigate('PINSetup'); // Force PIN creation
} else {
  // Returning user
  navigation.navigate('PINLock'); // → Unlock → Welcome
}
```

### Phase 4: Migration Strategy

For existing users with multiple profiles:

**Option A: Keep Active Profile Only**
```javascript
// On first launch after update:
const profiles = await AsyncStorage.getItem('@veilpath_user_profiles');
const activeID = await AsyncStorage.getItem('@veilpath_active_profile');

if (profiles && profiles.length > 1) {
  // Migrate active profile only
  const activeProfile = profiles.find(p => p.id === activeID);
  await AsyncStorage.setItem('@veilpath_user_profile', JSON.stringify(activeProfile));

  // Show warning
  Alert.alert(
    'Single Profile Mode',
    'Lunatiq now supports one profile per device. Your active profile has been preserved. Other profiles have been removed.',
    [{ text: 'OK' }]
  );

  // Clean up old keys
  await AsyncStorage.removeItem('@veilpath_user_profiles');
  await AsyncStorage.removeItem('@veilpath_active_profile');
}
```

**Option B: Let User Choose**
```javascript
// Show one-time migration screen
if (profiles && profiles.length > 1) {
  navigation.navigate('ProfileMigration', { profiles });
  // Let user pick which profile to keep
  // Others are discarded (or offered export)
}
```

## Benefits Summary

### For Users
- ✅ Simpler onboarding (no profile switching confusion)
- ✅ Better privacy (PIN-protected personal data)
- ✅ Cloud sync "just works" (one profile = one cloud account)
- ✅ Biometric unlock (optional convenience)

### For Development
- ✅ Simpler codebase (no profile management logic)
- ✅ Fewer edge cases (no "active profile undefined" bugs)
- ✅ Cleaner data model (1:1 device:profile mapping)
- ✅ Better security (forced PIN, no profile hopping)

### For Security
- ✅ Harder to exploit (can't create throwaway profiles)
- ✅ Ban enforcement (banning device = banning user)
- ✅ Subscription integrity (can't share across profiles)
- ✅ Device lock protection (PIN required)

## User Communication

**In-App Message** (for existing users):
```
🔒 LUNATIQ IS NOW MORE SECURE

To protect your personal readings and journal, Lunatiq now:
- Uses ONE profile per device (no more switching)
- Requires a PIN to unlock the app

Your active profile has been preserved.
Let's set up your PIN now to keep your data private.

[Set Up PIN]
```

**App Store Update Notes**:
```
Version X.X.X - Enhanced Privacy & Security

- Single profile mode for better privacy
- Mandatory PIN protection
- Automatic cloud backup (Premium)
- Improved anti-cheat protection

Your active profile has been automatically migrated.
Other profiles can be exported before updating if needed.
```

## FAQ

**Q: What if I lose my PIN?**
A: Security questions or email reset (to be implemented).

**Q: Can I change my PIN?**
A: Yes, in Settings → Security → Change PIN.

**Q: What if someone gets my phone?**
A: Without your PIN, they can't access your readings or journal.

**Q: Can I have multiple profiles on different devices?**
A: No. One device = one profile. But you can use different devices with different accounts.

**Q: What happened to my other profiles?**
A: Only your active profile was migrated. Others were removed for security reasons.

---

**TL;DR**: One device, one profile, one PIN. Simpler, more secure, harder to exploit.
