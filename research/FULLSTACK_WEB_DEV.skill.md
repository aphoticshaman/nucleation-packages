# FULLSTACK_WEB_DEV.skill.md

## Full-Stack Engineering: Web, Mobile, and Beyond

**Version**: 1.0
**Domain**: Full-Stack Development, React/Next.js, React Native, APIs, Databases
**Prerequisites**: CODING_BEST_PRACTICES skill
**Output**: Production-ready web and mobile applications

---

## 1. EXECUTIVE SUMMARY

Full-stack development is about understanding the complete picture—from user interaction to database and back. This skill covers patterns for building modern web applications, mobile apps, and the APIs that connect them.

**Core Principle**: Own the whole stack. Understand every layer. Optimize for the user.

---

## 2. ARCHITECTURE PATTERNS

### 2.1 Modern Web Stack

```
RECOMMENDED STACK (2024):
├── Frontend: React/Next.js + TypeScript
├── Mobile: React Native/Expo
├── API: Next.js API Routes or separate Node.js
├── Database: PostgreSQL + Prisma
├── Cache: Redis
├── Auth: NextAuth.js or Clerk
├── Hosting: Vercel + Railway/Supabase
├── CDN: Cloudflare
└── Monitoring: Vercel Analytics + Sentry

WHY THIS STACK:
├── TypeScript everywhere: One language, type safety
├── React ecosystem: Components work web and mobile
├── Vercel: Best Next.js hosting, automatic scaling
├── PostgreSQL: Reliable, feature-rich, free tier options
├── Prisma: Type-safe database access
```

### 2.2 Application Architecture

```typescript
// CLEAN ARCHITECTURE FOR FULL-STACK

// Layer 1: Domain (pure business logic)
// src/domain/
interface User {
  id: string;
  email: string;
  name: string;
}

interface UserRepository {
  findById(id: string): Promise<User | null>;
  save(user: User): Promise<User>;
}

// Layer 2: Application (use cases)
// src/application/
class CreateUserUseCase {
  constructor(
    private userRepo: UserRepository,
    private emailService: EmailService,
  ) {}

  async execute(input: CreateUserInput): Promise<User> {
    const user = await this.userRepo.save({
      id: generateId(),
      email: input.email,
      name: input.name,
    });

    await this.emailService.sendWelcome(user.email);
    return user;
  }
}

// Layer 3: Infrastructure (implementations)
// src/infrastructure/
class PrismaUserRepository implements UserRepository {
  constructor(private prisma: PrismaClient) {}

  async findById(id: string): Promise<User | null> {
    return this.prisma.user.findUnique({ where: { id } });
  }
}

// Layer 4: Presentation (API/UI)
// src/pages/api/users.ts
export default async function handler(req, res) {
  const useCase = new CreateUserUseCase(
    new PrismaUserRepository(prisma),
    new ResendEmailService(),
  );

  const user = await useCase.execute(req.body);
  res.json(user);
}
```

### 2.3 Feature-Based Structure

```
PROJECT STRUCTURE:
src/
├── features/          # Feature modules
│   ├── auth/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── api/
│   │   └── types.ts
│   ├── users/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── api/
│   │   └── types.ts
│   └── orders/
├── shared/            # Shared utilities
│   ├── components/
│   ├── hooks/
│   ├── utils/
│   └── types/
├── lib/               # External integrations
│   ├── prisma.ts
│   ├── redis.ts
│   └── stripe.ts
└── pages/             # Next.js routes
    ├── api/
    └── [...routes]
```

---

## 3. REACT/NEXT.JS PATTERNS

### 3.1 Component Patterns

```typescript
// COMPOUND COMPONENT PATTERN
const Card = ({ children }: { children: React.ReactNode }) => {
  return <div className="card">{children}</div>;
};

Card.Header = ({ children }: { children: React.ReactNode }) => {
  return <div className="card-header">{children}</div>;
};

Card.Body = ({ children }: { children: React.ReactNode }) => {
  return <div className="card-body">{children}</div>;
};

// Usage
<Card>
  <Card.Header>Title</Card.Header>
  <Card.Body>Content</Card.Body>
</Card>

// RENDER PROPS PATTERN
interface DataFetcherProps<T> {
  url: string;
  children: (data: T | null, loading: boolean, error: Error | null) => React.ReactNode;
}

function DataFetcher<T>({ url, children }: DataFetcherProps<T>) {
  const { data, loading, error } = useFetch<T>(url);
  return <>{children(data, loading, error)}</>;
}

// CONTAINER/PRESENTER PATTERN
// Container (logic)
function UserListContainer() {
  const { users, loading, error } = useUsers();

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;

  return <UserList users={users} />;
}

// Presenter (UI only)
function UserList({ users }: { users: User[] }) {
  return (
    <ul>
      {users.map(user => (
        <UserCard key={user.id} user={user} />
      ))}
    </ul>
  );
}
```

### 3.2 Custom Hooks

```typescript
// DATA FETCHING HOOK
function useQuery<T>(
  key: string,
  fetcher: () => Promise<T>,
  options: QueryOptions = {}
) {
  const [state, setState] = useState<QueryState<T>>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;

    const fetch = async () => {
      try {
        const data = await fetcher();
        if (!cancelled) {
          setState({ data, loading: false, error: null });
        }
      } catch (error) {
        if (!cancelled) {
          setState({ data: null, loading: false, error: error as Error });
        }
      }
    };

    fetch();

    return () => {
      cancelled = true;
    };
  }, [key]);

  return state;
}

// MUTATION HOOK
function useMutation<TData, TVariables>(
  mutationFn: (variables: TVariables) => Promise<TData>
) {
  const [state, setState] = useState<MutationState<TData>>({
    data: null,
    loading: false,
    error: null,
  });

  const mutate = useCallback(async (variables: TVariables) => {
    setState({ data: null, loading: true, error: null });

    try {
      const data = await mutationFn(variables);
      setState({ data, loading: false, error: null });
      return data;
    } catch (error) {
      setState({ data: null, loading: false, error: error as Error });
      throw error;
    }
  }, [mutationFn]);

  return { ...state, mutate };
}

// LOCAL STORAGE HOOK
function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    if (typeof window === 'undefined') return initialValue;

    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      return initialValue;
    }
  });

  const setValue = (value: T | ((val: T) => T)) => {
    const valueToStore = value instanceof Function ? value(storedValue) : value;
    setStoredValue(valueToStore);
    window.localStorage.setItem(key, JSON.stringify(valueToStore));
  };

  return [storedValue, setValue] as const;
}
```

### 3.3 Server Components (Next.js 13+)

```typescript
// SERVER COMPONENT (default in app/ directory)
// app/users/page.tsx
async function UsersPage() {
  // This runs on the server - can directly access DB
  const users = await prisma.user.findMany({
    orderBy: { createdAt: 'desc' },
    take: 10,
  });

  return (
    <div>
      <h1>Users</h1>
      <UserList users={users} />
    </div>
  );
}

// CLIENT COMPONENT (needs interactivity)
// app/users/UserList.tsx
'use client';

import { useState } from 'react';

export function UserList({ users }: { users: User[] }) {
  const [filter, setFilter] = useState('');

  const filtered = users.filter(u =>
    u.name.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div>
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter users..."
      />
      <ul>
        {filtered.map(user => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  );
}

// WHEN TO USE EACH:
// Server: Data fetching, no interactivity, SEO-critical
// Client: State, effects, event handlers, browser APIs
```

---

## 4. API DESIGN

### 4.1 RESTful API Patterns

```typescript
// API ROUTE STRUCTURE (Next.js)
// pages/api/users/index.ts - Collection
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  switch (req.method) {
    case 'GET':
      const users = await prisma.user.findMany();
      return res.json(users);

    case 'POST':
      const user = await prisma.user.create({
        data: req.body,
      });
      return res.status(201).json(user);

    default:
      res.setHeader('Allow', ['GET', 'POST']);
      return res.status(405).end();
  }
}

// pages/api/users/[id].ts - Individual resource
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { id } = req.query;

  switch (req.method) {
    case 'GET':
      const user = await prisma.user.findUnique({ where: { id: String(id) } });
      if (!user) return res.status(404).json({ error: 'Not found' });
      return res.json(user);

    case 'PATCH':
      const updated = await prisma.user.update({
        where: { id: String(id) },
        data: req.body,
      });
      return res.json(updated);

    case 'DELETE':
      await prisma.user.delete({ where: { id: String(id) } });
      return res.status(204).end();

    default:
      res.setHeader('Allow', ['GET', 'PATCH', 'DELETE']);
      return res.status(405).end();
  }
}
```

### 4.2 API Middleware

```typescript
// MIDDLEWARE PATTERN
type Middleware = (
  req: NextApiRequest,
  res: NextApiResponse,
  next: () => Promise<void>
) => Promise<void>;

function withMiddleware(...middlewares: Middleware[]) {
  return (handler: NextApiHandler): NextApiHandler => {
    return async (req, res) => {
      let index = 0;

      const next = async () => {
        if (index < middlewares.length) {
          const middleware = middlewares[index++];
          await middleware(req, res, next);
        } else {
          await handler(req, res);
        }
      };

      await next();
    };
  };
}

// COMMON MIDDLEWARES
const withAuth: Middleware = async (req, res, next) => {
  const token = req.headers.authorization?.replace('Bearer ', '');

  if (!token) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  try {
    const user = await verifyToken(token);
    req.user = user;
    await next();
  } catch {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

const withLogging: Middleware = async (req, res, next) => {
  const start = Date.now();
  await next();
  const duration = Date.now() - start;
  console.log(`${req.method} ${req.url} - ${res.statusCode} (${duration}ms)`);
};

// USAGE
export default withMiddleware(
  withLogging,
  withAuth,
)(async function handler(req, res) {
  // Handler has access to req.user
  res.json({ user: req.user });
});
```

### 4.3 Error Handling

```typescript
// API ERROR CLASSES
class ApiError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public code?: string,
  ) {
    super(message);
  }
}

class NotFoundError extends ApiError {
  constructor(resource: string) {
    super(404, `${resource} not found`, 'NOT_FOUND');
  }
}

class ValidationError extends ApiError {
  constructor(
    message: string,
    public errors: Record<string, string[]>
  ) {
    super(400, message, 'VALIDATION_ERROR');
  }
}

// ERROR HANDLING WRAPPER
function withErrorHandling(handler: NextApiHandler): NextApiHandler {
  return async (req, res) => {
    try {
      await handler(req, res);
    } catch (error) {
      if (error instanceof ApiError) {
        return res.status(error.statusCode).json({
          error: {
            message: error.message,
            code: error.code,
            ...(error instanceof ValidationError && { errors: error.errors }),
          },
        });
      }

      // Unexpected error
      console.error('Unhandled error:', error);
      return res.status(500).json({
        error: {
          message: 'Internal server error',
          code: 'INTERNAL_ERROR',
        },
      });
    }
  };
}
```

---

## 5. DATABASE PATTERNS

### 5.1 Prisma Best Practices

```typescript
// PRISMA SCHEMA (prisma/schema.prisma)
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String?
  orders    Order[]
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@index([email])
}

model Order {
  id        String      @id @default(cuid())
  userId    String
  user      User        @relation(fields: [userId], references: [id])
  items     OrderItem[]
  total     Decimal     @db.Decimal(10, 2)
  status    OrderStatus @default(PENDING)
  createdAt DateTime    @default(now())

  @@index([userId])
  @@index([status])
}

enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
}

// PRISMA CLIENT SINGLETON
// lib/prisma.ts
import { PrismaClient } from '@prisma/client';

const globalForPrisma = global as unknown as { prisma: PrismaClient };

export const prisma =
  globalForPrisma.prisma ||
  new PrismaClient({
    log: process.env.NODE_ENV === 'development'
      ? ['query', 'error', 'warn']
      : ['error'],
  });

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}

// QUERY PATTERNS
// Efficient includes
const userWithOrders = await prisma.user.findUnique({
  where: { id },
  include: {
    orders: {
      where: { status: 'PENDING' },
      orderBy: { createdAt: 'desc' },
      take: 10,
    },
  },
});

// Transaction
const [user, order] = await prisma.$transaction([
  prisma.user.update({ where: { id }, data: { balance: { decrement: 100 } } }),
  prisma.order.create({ data: { userId: id, total: 100 } }),
]);

// Batch operations
const users = await prisma.user.createMany({
  data: usersToCreate,
  skipDuplicates: true,
});
```

### 5.2 Query Optimization

```typescript
// N+1 PROBLEM SOLUTION
// BAD: N+1 queries
const orders = await prisma.order.findMany();
for (const order of orders) {
  order.user = await prisma.user.findUnique({ where: { id: order.userId } });
}

// GOOD: Single query with include
const orders = await prisma.order.findMany({
  include: { user: true },
});

// PAGINATION
async function getPaginatedUsers(page: number, pageSize: number = 20) {
  const [users, total] = await prisma.$transaction([
    prisma.user.findMany({
      skip: (page - 1) * pageSize,
      take: pageSize,
      orderBy: { createdAt: 'desc' },
    }),
    prisma.user.count(),
  ]);

  return {
    data: users,
    pagination: {
      page,
      pageSize,
      total,
      totalPages: Math.ceil(total / pageSize),
    },
  };
}

// CURSOR-BASED PAGINATION (better for large datasets)
async function getCursorPaginatedUsers(cursor?: string, take: number = 20) {
  const users = await prisma.user.findMany({
    take: take + 1, // Fetch one extra to check if there's more
    ...(cursor && {
      cursor: { id: cursor },
      skip: 1, // Skip the cursor
    }),
    orderBy: { createdAt: 'desc' },
  });

  const hasMore = users.length > take;
  const data = hasMore ? users.slice(0, -1) : users;

  return {
    data,
    nextCursor: hasMore ? data[data.length - 1].id : null,
  };
}
```

---

## 6. REACT NATIVE / EXPO

### 6.1 Expo Setup

```typescript
// App.tsx
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const Stack = createNativeStackNavigator();
const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen name="Details" component={DetailsScreen} />
        </Stack.Navigator>
        <StatusBar style="auto" />
      </NavigationContainer>
    </QueryClientProvider>
  );
}

// SHARED COMPONENTS (web + mobile)
// components/Button.tsx
import { Platform, Pressable, Text, StyleSheet } from 'react-native';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary';
}

export function Button({ title, onPress, variant = 'primary' }: ButtonProps) {
  return (
    <Pressable
      style={({ pressed }) => [
        styles.button,
        styles[variant],
        pressed && styles.pressed,
      ]}
      onPress={onPress}
    >
      <Text style={[styles.text, styles[`${variant}Text`]]}>{title}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    alignItems: 'center',
  },
  primary: {
    backgroundColor: '#8B5CF6',
  },
  secondary: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#8B5CF6',
  },
  pressed: {
    opacity: 0.8,
  },
  text: {
    fontSize: 16,
    fontWeight: '600',
  },
  primaryText: {
    color: '#FFFFFF',
  },
  secondaryText: {
    color: '#8B5CF6',
  },
});
```

### 6.2 Platform-Specific Code

```typescript
// PLATFORM-SPECIFIC FILES
// Button.tsx (default)
// Button.ios.tsx (iOS override)
// Button.android.tsx (Android override)
// Button.web.tsx (Web override)

// OR INLINE PLATFORM CHECKS
import { Platform } from 'react-native';

const styles = StyleSheet.create({
  container: {
    padding: Platform.select({
      ios: 20,
      android: 16,
      web: 24,
      default: 16,
    }),
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
      },
      android: {
        elevation: 4,
      },
    }),
  },
});

// CONDITIONAL RENDERING
function PaymentButton() {
  if (Platform.OS === 'ios') {
    return <ApplePayButton />;
  }
  if (Platform.OS === 'android') {
    return <GooglePayButton />;
  }
  return <StripeButton />;
}
```

---

## 7. STATE MANAGEMENT

### 7.1 Zustand (Recommended)

```typescript
// stores/useStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AppState {
  // State
  user: User | null;
  cart: CartItem[];
  theme: 'light' | 'dark';

  // Actions
  setUser: (user: User | null) => void;
  addToCart: (item: CartItem) => void;
  removeFromCart: (itemId: string) => void;
  clearCart: () => void;
  toggleTheme: () => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      // Initial state
      user: null,
      cart: [],
      theme: 'dark',

      // Actions
      setUser: (user) => set({ user }),

      addToCart: (item) =>
        set((state) => ({
          cart: [...state.cart, item],
        })),

      removeFromCart: (itemId) =>
        set((state) => ({
          cart: state.cart.filter((item) => item.id !== itemId),
        })),

      clearCart: () => set({ cart: [] }),

      toggleTheme: () =>
        set((state) => ({
          theme: state.theme === 'light' ? 'dark' : 'light',
        })),
    }),
    {
      name: 'app-storage',
      partialize: (state) => ({
        theme: state.theme,
        cart: state.cart,
      }),
    }
  )
);

// USAGE
function CartButton() {
  const { cart, addToCart } = useStore();

  return (
    <button onClick={() => addToCart(item)}>
      Add to Cart ({cart.length})
    </button>
  );
}

// SELECTORS (prevent unnecessary re-renders)
const cartCount = useStore((state) => state.cart.length);
const isLoggedIn = useStore((state) => state.user !== null);
```

### 7.2 React Query for Server State

```typescript
// hooks/useUsers.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// Fetch users
export function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: () => fetch('/api/users').then((res) => res.json()),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Fetch single user
export function useUser(id: string) {
  return useQuery({
    queryKey: ['users', id],
    queryFn: () => fetch(`/api/users/${id}`).then((res) => res.json()),
    enabled: !!id, // Don't run if no id
  });
}

// Create user mutation
export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateUserInput) =>
      fetch('/api/users', {
        method: 'POST',
        body: JSON.stringify(data),
      }).then((res) => res.json()),

    onSuccess: () => {
      // Invalidate users list to refetch
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}

// USAGE
function UsersList() {
  const { data: users, isLoading, error } = useUsers();
  const createUser = useCreateUser();

  if (isLoading) return <Spinner />;
  if (error) return <Error error={error} />;

  return (
    <div>
      {users.map((user) => (
        <UserCard key={user.id} user={user} />
      ))}
      <button
        onClick={() => createUser.mutate({ name: 'New User', email: '...' })}
        disabled={createUser.isPending}
      >
        Add User
      </button>
    </div>
  );
}
```

---

## 8. PERFORMANCE OPTIMIZATION

### 8.1 React Performance

```typescript
// MEMOIZATION
const MemoizedComponent = React.memo(function ExpensiveComponent({ data }) {
  return <div>{/* expensive render */}</div>;
});

// useMemo for expensive calculations
function DataTable({ data }) {
  const sortedData = useMemo(
    () => data.sort((a, b) => a.date - b.date),
    [data]
  );

  return <Table data={sortedData} />;
}

// useCallback for stable function references
function Parent() {
  const handleClick = useCallback((id: string) => {
    // handle click
  }, [/* dependencies */]);

  return <Child onClick={handleClick} />;
}

// VIRTUALIZATION for long lists
import { FixedSizeList } from 'react-window';

function VirtualizedList({ items }) {
  const Row = ({ index, style }) => (
    <div style={style}>{items[index].name}</div>
  );

  return (
    <FixedSizeList
      height={400}
      width="100%"
      itemCount={items.length}
      itemSize={50}
    >
      {Row}
    </FixedSizeList>
  );
}

// CODE SPLITTING
const HeavyComponent = dynamic(() => import('./HeavyComponent'), {
  loading: () => <Spinner />,
  ssr: false, // Disable server-side rendering if needed
});
```

### 8.2 Next.js Optimization

```typescript
// IMAGE OPTIMIZATION
import Image from 'next/image';

<Image
  src="/hero.jpg"
  alt="Hero image"
  width={1200}
  height={600}
  priority // Above the fold
  placeholder="blur"
  blurDataURL="data:image/jpeg;base64,..."
/>

// FONT OPTIMIZATION
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

// STATIC GENERATION
export async function getStaticProps() {
  const posts = await fetchPosts();
  return {
    props: { posts },
    revalidate: 60, // ISR: regenerate every 60 seconds
  };
}

// STATIC PATHS
export async function getStaticPaths() {
  const posts = await fetchPosts();
  return {
    paths: posts.map((post) => ({ params: { slug: post.slug } })),
    fallback: 'blocking', // Generate new pages on demand
  };
}

// METADATA
export const metadata = {
  title: 'Page Title',
  description: 'Page description',
  openGraph: {
    images: ['/og-image.png'],
  },
};
```

---

## 9. TESTING

### 9.1 Component Testing

```typescript
// __tests__/UserCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { UserCard } from '../components/UserCard';

describe('UserCard', () => {
  const mockUser = {
    id: '1',
    name: 'Test User',
    email: 'test@example.com',
  };

  it('renders user information', () => {
    render(<UserCard user={mockUser} />);

    expect(screen.getByText('Test User')).toBeInTheDocument();
    expect(screen.getByText('test@example.com')).toBeInTheDocument();
  });

  it('calls onEdit when edit button clicked', () => {
    const onEdit = jest.fn();
    render(<UserCard user={mockUser} onEdit={onEdit} />);

    fireEvent.click(screen.getByRole('button', { name: /edit/i }));

    expect(onEdit).toHaveBeenCalledWith(mockUser.id);
  });
});

// API ROUTE TESTING
import { createMocks } from 'node-mocks-http';
import handler from '../pages/api/users';

describe('/api/users', () => {
  it('returns users on GET', async () => {
    const { req, res } = createMocks({ method: 'GET' });

    await handler(req, res);

    expect(res._getStatusCode()).toBe(200);
    expect(JSON.parse(res._getData())).toHaveLength(10);
  });

  it('creates user on POST', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      body: { name: 'New User', email: 'new@example.com' },
    });

    await handler(req, res);

    expect(res._getStatusCode()).toBe(201);
    expect(JSON.parse(res._getData()).id).toBeDefined();
  });
});
```

---

## 10. IMPLEMENTATION CHECKLIST

### Project Setup:
- [ ] TypeScript configured
- [ ] ESLint + Prettier setup
- [ ] Prisma configured
- [ ] Environment variables managed
- [ ] CI/CD pipeline ready

### Frontend:
- [ ] Component library chosen
- [ ] Routing configured
- [ ] State management set up
- [ ] API client implemented
- [ ] Error boundaries added

### Backend:
- [ ] API routes structured
- [ ] Authentication implemented
- [ ] Error handling middleware
- [ ] Logging configured
- [ ] Rate limiting added

### Performance:
- [ ] Images optimized
- [ ] Code splitting implemented
- [ ] Caching strategy defined
- [ ] Bundle size monitored
- [ ] Core Web Vitals tracked

---

**Remember**: Full-stack is about understanding the whole picture—from pixels to database rows. Master each layer, but always optimize for what the user experiences.

Build end-to-end. Ship with confidence.
