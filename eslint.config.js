import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import security from 'eslint-plugin-security';

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    plugins: {
      security,
    },
    languageOptions: {
      parserOptions: {
        projectService: {
          allowDefaultProject: ['packages/*/test/*.ts'],
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Security rules - keep strict for enterprise/defense
      'security/detect-object-injection': 'warn',
      'security/detect-non-literal-regexp': 'warn',
      'security/detect-unsafe-regex': 'error',
      'security/detect-buffer-noassert': 'error',
      'security/detect-eval-with-expression': 'error',
      'security/detect-no-csrf-before-method-override': 'error',
      'security/detect-possible-timing-attacks': 'warn',

      // TypeScript - pragmatic for production stability
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/no-non-null-assertion': 'warn',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': 'error',

      // Code quality
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      eqeqeq: ['error', 'always'],
      'no-var': 'error',
      'prefer-const': 'error',
      'no-throw-literal': 'error',
    },
  },
  {
    files: ['**/*.test.ts', '**/*.spec.ts', '**/test/**/*.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      'no-console': 'off',
    },
  },
  {
    ignores: [
      'node_modules/**',
      '**/node_modules/**',
      'dist/**',
      '**/dist/**',
      'coverage/**',
      'harness/**',
      // Apps directory (separate projects with own configs)
      'apps/**',
      // Legacy JS packages - will migrate to TypeScript later
      'packages/regime-shift/**',
      'packages/threat-pulse/**',
      'packages/churn-harbinger/**',
      'packages/org-canary/**',
      'packages/supply-sentinel/**',
      'packages/sensor-shift/**',
      'packages/crowd-phase/**',
      'packages/patient-drift/**',
      'packages/match-pulse/**',
      'packages/market-canary/**',
      'packages/nucleation/**',
      // Data source packages (need separate security review for regex patterns)
      'packages/sources/**',
      'packages/compliance/**',
      // Web package JS config files
      'packages/web/*.js',
      'packages/web/public/**',
      // Supabase edge functions (Deno runtime)
      'supabase/**',
      '*.js',
      '*.d.ts',
      'vitest.config.ts',
      '**/vitest.config.ts',
    ],
  }
);
