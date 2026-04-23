import { useState, useEffect } from 'react';
import { useQuery, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { 
  createRootRoute, 
  createRoute, 
  createRouter, 
  RouterProvider, 
  Link, 
  Outlet, 
  useParams
} from '@tanstack/react-router';
import { Users, Trophy, Sun, Moon, Monitor, Hash, ArrowLeft, Star } from 'lucide-react';
import { getDimensions, getTopPlayers, getHardestBeatmaps, getBeatmap, getBeatmapScores } from './api';
import type { Player, Beatmap, Score } from './api';
import { cn } from './lib/utils';

const queryClient = new QueryClient();

type Theme = 'light' | 'dark' | 'system';

// --- Router Setup ---

const rootRoute = createRootRoute({
  component: RootComponent,
});

function RootComponent() {
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem('theme') as Theme) || 'system';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const applyTheme = () => {
      const isDark = theme === 'dark' || (theme === 'system' && mediaQuery.matches);
      root.classList.toggle('dark', isDark);
    };

    applyTheme();
    localStorage.setItem('theme', theme);

    if (theme === 'system') {
      mediaQuery.addEventListener('change', applyTheme);
      return () => mediaQuery.removeEventListener('change', applyTheme);
    }
  }, [theme]);

  return (
    <div className="min-h-screen bg-white dark:bg-[#0a0a0a] text-zinc-900 dark:text-zinc-100 transition-colors duration-200 font-sans selection:bg-zinc-200 dark:selection:bg-zinc-800">
      <nav className="sticky top-0 z-50 border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-[#0a0a0a]/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-zinc-900 dark:bg-zinc-100 rounded flex items-center justify-center">
              <Hash className="w-5 h-5 text-white dark:text-black" />
            </div>
            <span className="font-semibold tracking-tight text-lg">statpp</span>
          </Link>

          <div className="flex items-center gap-2">
            {(['light', 'dark', 'system'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTheme(t)}
                className={cn(
                  "p-2 rounded-md transition-colors",
                  theme === t 
                    ? "bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100" 
                    : "text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
                )}
              >
                {t === 'light' && <Sun className="w-4 h-4" />}
                {t === 'dark' && <Moon className="w-4 h-4" />}
                {t === 'system' && <Monitor className="w-4 h-4" />}
              </button>
            ))}
          </div>
        </div>
      </nav>
      <Outlet />
    </div>
  );
}

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: Dashboard,
});

const beatmapRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/beatmap/$beatmapId',
  component: BeatmapPage,
});

const routeTree = rootRoute.addChildren([indexRoute, beatmapRoute]);
const router = createRouter({ routeTree });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}

// --- Components ---

function Dashboard() {
  const [dim, setDim] = useState(0);
  
  const { data: dimensions } = useQuery({
    queryKey: ['dimensions'],
    queryFn: getDimensions,
  });

  const { data: players, isLoading: playersLoading } = useQuery({
    queryKey: ['players', dim],
    queryFn: () => getTopPlayers(dim),
    enabled: dimensions !== undefined,
  });

  const { data: beatmaps, isLoading: beatmapsLoading } = useQuery({
    queryKey: ['beatmaps', dim],
    queryFn: () => getHardestBeatmaps(dim),
    enabled: dimensions !== undefined,
  });

  return (
    <main className="max-w-7xl mx-auto px-6 py-10">
      <header className="mb-12">
        <div className="flex flex-col gap-1 mb-8">
          <h1 className="text-3xl font-bold tracking-tight">Dimensions Explorer</h1>
          <p className="text-zinc-500 dark:text-zinc-400">Slice and dice n-dimensional rating data with sub-second precision.</p>
        </div>
        
        {dimensions !== undefined && (
          <div className="flex flex-wrap gap-2">
            {Array.from({ length: dimensions }).map((_, i) => (
              <button
                key={i}
                onClick={() => setDim(i)}
                className={cn(
                  "px-4 py-1.5 rounded-full text-sm font-medium transition-all border",
                  dim === i 
                    ? "bg-zinc-900 dark:bg-zinc-100 text-white dark:text-black border-transparent" 
                    : "bg-transparent border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400 hover:border-zinc-400 dark:hover:border-zinc-600"
                )}
              >
                Dimension {i}
              </button>
            ))}
          </div>
        )}
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        <section className="space-y-4">
          <div className="flex items-center gap-2 px-1">
            <Users className="w-5 h-5 text-zinc-400" />
            <h2 className="text-xl font-semibold">Top Players</h2>
          </div>
          <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-zinc-50/50 dark:bg-zinc-900/30">
            <table className="w-full text-sm text-left border-collapse">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-transparent">
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">ID</th>
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Rating</th>
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400 text-right">User</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                {playersLoading ? (
                  [...Array(5)].map((_, i) => (
                    <tr key={i} className="animate-pulse">
                      <td colSpan={3} className="px-4 py-4 h-12 bg-zinc-100/50 dark:bg-zinc-800/20" />
                    </tr>
                  ))
                ) : players?.map((p: Player) => (
                  <tr key={p.id} className="group hover:bg-white dark:hover:bg-zinc-800/40 transition-colors">
                    <td className="px-4 py-3 font-mono text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100 transition-colors">{p.id}</td>
                    <td className="px-4 py-3 font-semibold tabular-nums">
                      {p.metadata.ratings[dim]?.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="text-xs text-zinc-400 font-mono truncate max-w-[120px] inline-block">
                        {p.metadata.username}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex items-center gap-2 px-1">
            <Trophy className="w-5 h-5 text-zinc-400" />
            <h2 className="text-xl font-semibold">Hardest Beatmaps</h2>
          </div>
          <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-zinc-50/50 dark:bg-zinc-900/30">
            <table className="w-full text-sm text-left border-collapse">
              <thead>
                <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-transparent">
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">ID</th>
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Difficulty</th>
                  <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400 text-right">Title</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                {beatmapsLoading ? (
                  [...Array(5)].map((_, i) => (
                    <tr key={i} className="animate-pulse">
                      <td colSpan={3} className="px-4 py-4 h-12 bg-zinc-100/50 dark:bg-zinc-800/20" />
                    </tr>
                  ))
                ) : beatmaps?.map((b: Beatmap) => (
                  <tr key={b.id} className="group hover:bg-white dark:hover:bg-zinc-800/40 transition-colors">
                    <td className="px-4 py-3 font-mono text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100 transition-colors">
                      <Link to="/beatmap/$beatmapId" params={{ beatmapId: b.id.toString() }} className="hover:underline">
                        {b.id}
                      </Link>
                    </td>
                    <td className="px-4 py-3 font-semibold tabular-nums">
                      {b.metadata.difficulties[dim]?.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Link to="/beatmap/$beatmapId" params={{ beatmapId: b.id.toString() }} className="text-xs text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 font-mono truncate max-w-[120px] inline-block transition-colors">
                        {b.metadata.title}
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}

function BeatmapPage() {
  const { beatmapId } = useParams({ from: '/beatmap/$beatmapId' });
  const [dim, setDim] = useState(0);

  const { data: beatmap, isLoading: beatmapLoading } = useQuery({
    queryKey: ['beatmap', beatmapId],
    queryFn: () => getBeatmap(parseInt(beatmapId)),
  });

  const { data: dimensions } = useQuery({
    queryKey: ['dimensions'],
    queryFn: getDimensions,
  });

  const { data: scores, isLoading: scoresLoading } = useQuery({
    queryKey: ['scores', beatmapId, dim],
    queryFn: () => getBeatmapScores(parseInt(beatmapId), dim),
    enabled: dimensions !== undefined,
  });

  return (
    <main className="max-w-7xl mx-auto px-6 py-10">
      <Link to="/" className="inline-flex items-center gap-2 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 mb-8 transition-colors group">
        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
        <span className="text-sm font-medium">Back to Explorer</span>
      </Link>

      {beatmapLoading ? (
        <div className="h-20 w-1/3 bg-zinc-100 dark:bg-zinc-800 animate-pulse rounded-lg mb-8" />
      ) : beatmap && (
        <header className="mb-12">
          <h1 className="text-4xl font-bold tracking-tight mb-2">{beatmap.metadata.title}</h1>
          <p className="text-zinc-500 dark:text-zinc-400 font-mono text-sm">Beatmap ID: {beatmap.id}</p>
          
          <div className="mt-8">
            <h3 className="text-sm font-medium text-zinc-400 mb-4 uppercase tracking-wider">Difficulties by Dimension</h3>
            <div className="flex flex-wrap gap-4">
              {beatmap.metadata.difficulties.map((d: number, i: number) => (
                <div key={i} className="bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 px-4 py-3 rounded-lg">
                  <div className="text-xs text-zinc-500 mb-1">Dim {i}</div>
                  <div className="text-lg font-bold tabular-nums">{d.toFixed(4)}</div>
                </div>
              ))}
            </div>
          </div>
        </header>
      )}

      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Star className="w-5 h-5 text-zinc-400" />
            <h2 className="text-xl font-semibold">Top Scores</h2>
          </div>

          {dimensions !== undefined && (
            <div className="flex gap-1">
              {Array.from({ length: dimensions }).map((_, i) => (
                <button
                  key={i}
                  onClick={() => setDim(i)}
                  className={cn(
                    "px-3 py-1 rounded-md text-xs font-medium transition-all",
                    dim === i 
                      ? "bg-zinc-900 dark:bg-zinc-100 text-white dark:text-black" 
                      : "bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100"
                  )}
                >
                  Dim {i}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="border border-zinc-200 dark:border-zinc-800 rounded-lg overflow-hidden bg-zinc-50/50 dark:bg-zinc-900/30">
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-transparent">
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Player ID</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Mod</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400">Score (Dim {dim})</th>
                <th className="px-4 py-3 font-medium text-zinc-500 dark:text-zinc-400 text-right">Accuracy</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
              {scoresLoading ? (
                [...Array(5)].map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    <td colSpan={4} className="px-4 py-4 h-12 bg-zinc-100/50 dark:bg-zinc-800/20" />
                  </tr>
                ))
              ) : scores?.map((s: Score, idx: number) => (
                <tr key={idx} className="group hover:bg-white dark:hover:bg-zinc-800/40 transition-colors">
                  <td className="px-4 py-3 font-mono text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-zinc-100 transition-colors">{s.player_id}</td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 rounded bg-zinc-200 dark:bg-zinc-800 text-[10px] font-bold uppercase tracking-wider">
                      {s.mod_str}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-semibold tabular-nums">
                    {s.metadata.scores[dim]?.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-zinc-400">
                    {(s.metadata.accuracy * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}

export default App;
