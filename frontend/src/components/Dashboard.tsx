import { useEffect, useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import {
  Brain, Users, SeparatorHorizontal, TrendingUp, Network,
  Trophy, Target, Eye, BarChart3, Activity
} from 'lucide-react'

type ModelData = {
  name: string
  type: string
  icon: string
  color: string
  description: string
  how_it_works: string
  strengths: string[]
  weaknesses: string[]
  metrics: {
    accuracy: number
    precision: number
    recall: number
    f1: number
    auc: number
  }
}

type DashboardData = {
  models: ModelData[]
  best_model: string
  dataset_info: {
    total_samples: number
    spam_count: number
    ham_count: number
    spam_percentage: number
    ham_percentage: number
  }
}

const iconMap: Record<string, React.ReactNode> = {
  'brain': <Brain className="h-8 w-8" />,
  'users': <Users className="h-8 w-8" />,
  'separator-horizontal': <SeparatorHorizontal className="h-8 w-8" />,
  'trending-up': <TrendingUp className="h-8 w-8" />,
  'network': <Network className="h-8 w-8" />,
}

function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-zinc-400">{label}</span>
        <span className="text-zinc-200 font-mono">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-1000"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

function ModelCard({ model, isBest }: { model: ModelData; isBest: boolean }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <Card
      className={`bg-zinc-900 border-zinc-800 hover:border-zinc-600 transition-all cursor-pointer ${isBest ? 'ring-2 ring-violet-500/50' : ''}`}
      onClick={() => setExpanded(!expanded)}
    >
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg" style={{ backgroundColor: `${model.color}20` }}>
              <div style={{ color: model.color }}>{iconMap[model.icon]}</div>
            </div>
            <div>
              <CardTitle className="text-lg text-zinc-100 flex items-center gap-2">
                {model.name}
                {isBest && (
                  <span className="text-xs bg-violet-500/20 text-violet-400 px-2 py-0.5 rounded-full flex items-center gap-1">
                    <Trophy className="h-3 w-3" /> Best
                  </span>
                )}
              </CardTitle>
              <p className="text-xs text-zinc-500">{model.type}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-zinc-100">{(model.metrics.f1 * 100).toFixed(1)}%</p>
            <p className="text-xs text-zinc-500">F1 Score</p>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-zinc-400">{model.description}</p>

        <div className="space-y-2">
          <MetricBar label="Accuracy" value={model.metrics.accuracy} color={model.color} />
          <MetricBar label="Precision" value={model.metrics.precision} color={model.color} />
          <MetricBar label="Recall" value={model.metrics.recall} color={model.color} />
          <MetricBar label="AUC" value={model.metrics.auc} color={model.color} />
        </div>

        {expanded && (
          <div className="space-y-4 pt-4 border-t border-zinc-800 animate-in fade-in">
            <div>
              <h4 className="text-sm font-semibold text-zinc-200 flex items-center gap-2 mb-2">
                <Eye className="h-4 w-4" /> How it works
              </h4>
              <p className="text-sm text-zinc-400">{model.how_it_works}</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-semibold text-green-400 mb-2">Strengths</h4>
                <ul className="space-y-1">
                  {model.strengths.map((s, i) => (
                    <li key={i} className="text-xs text-zinc-400 flex items-start gap-1">
                      <span className="text-green-500 mt-0.5">+</span> {s}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h4 className="text-sm font-semibold text-red-400 mb-2">Weaknesses</h4>
                <ul className="space-y-1">
                  {model.weaknesses.map((w, i) => (
                    <li key={i} className="text-xs text-zinc-400 flex items-start gap-1">
                      <span className="text-red-500 mt-0.5">-</span> {w}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        <p className="text-xs text-zinc-600 text-center">
          {expanded ? 'Click to collapse' : 'Click to learn more'}
        </p>
      </CardContent>
    </Card>
  )
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch('http://localhost:8000/models')
      .then(res => res.json())
      .then(setData)
      .catch(() => setError('Could not load dashboard data. Make sure the API is running.'))
  }, [])

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <p className="text-red-400">{error}</p>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
        <p className="text-zinc-400">Loading dashboard...</p>
      </div>
    )
  }

  const sortedModels = [...data.models].sort((a, b) => b.metrics.f1 - a.metrics.f1)

  return (
    <div className="space-y-8">
      {/* Dataset Overview */}
      <div>
        <h2 className="text-xl font-bold text-zinc-100 mb-4 flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-violet-500" /> Dataset Overview
        </h2>
        <div className="grid grid-cols-3 gap-4">
          <Card className="bg-zinc-900 border-zinc-800 text-center">
            <CardContent className="pt-6">
              <p className="text-3xl font-bold text-zinc-100">{data.dataset_info.total_samples.toLocaleString()}</p>
              <p className="text-sm text-zinc-500">Total Messages</p>
            </CardContent>
          </Card>
          <Card className="bg-zinc-900 border-zinc-800 text-center">
            <CardContent className="pt-6">
              <p className="text-3xl font-bold text-green-400">{data.dataset_info.ham_percentage}%</p>
              <p className="text-sm text-zinc-500">Ham ({data.dataset_info.ham_count.toLocaleString()})</p>
            </CardContent>
          </Card>
          <Card className="bg-zinc-900 border-zinc-800 text-center">
            <CardContent className="pt-6">
              <p className="text-3xl font-bold text-red-400">{data.dataset_info.spam_percentage}%</p>
              <p className="text-sm text-zinc-500">Spam ({data.dataset_info.spam_count.toLocaleString()})</p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div>
        <h2 className="text-xl font-bold text-zinc-100 mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 text-violet-500" /> Model Comparison
        </h2>
        <Card className="bg-zinc-900 border-zinc-800 overflow-hidden">
          <CardContent className="p-0">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800">
                  <th className="text-left p-4 text-zinc-400 font-medium">Model</th>
                  <th className="text-right p-4 text-zinc-400 font-medium">Accuracy</th>
                  <th className="text-right p-4 text-zinc-400 font-medium">Precision</th>
                  <th className="text-right p-4 text-zinc-400 font-medium">Recall</th>
                  <th className="text-right p-4 text-zinc-400 font-medium">F1</th>
                  <th className="text-right p-4 text-zinc-400 font-medium">AUC</th>
                </tr>
              </thead>
              <tbody>
                {sortedModels.map((model) => (
                  <tr
                    key={model.name}
                    className={`border-b border-zinc-800/50 hover:bg-zinc-800/30 ${model.name === data.best_model ? 'bg-violet-500/5' : ''}`}
                  >
                    <td className="p-4 text-zinc-100 font-medium flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: model.color }} />
                      {model.name}
                      {model.name === data.best_model && (
                        <Trophy className="h-3 w-3 text-violet-400" />
                      )}
                    </td>
                    <td className="text-right p-4 text-zinc-300 font-mono">{(model.metrics.accuracy * 100).toFixed(1)}%</td>
                    <td className="text-right p-4 text-zinc-300 font-mono">{(model.metrics.precision * 100).toFixed(1)}%</td>
                    <td className="text-right p-4 text-zinc-300 font-mono">{(model.metrics.recall * 100).toFixed(1)}%</td>
                    <td className="text-right p-4 text-zinc-300 font-mono font-bold">{(model.metrics.f1 * 100).toFixed(1)}%</td>
                    <td className="text-right p-4 text-zinc-300 font-mono">{(model.metrics.auc * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>

      {/* ROC Curve */}
      <div>
        <h2 className="text-xl font-bold text-zinc-100 mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-violet-500" /> ROC Curve Comparison
        </h2>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="pt-6 flex justify-center">
            <img
              src="http://localhost:8000/static/roc_curve.png"
              alt="ROC Curve Comparison"
              className="max-w-full rounded-lg"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none'
              }}
            />
          </CardContent>
        </Card>
      </div>

      {/* Model Cards */}
      <div>
        <h2 className="text-xl font-bold text-zinc-100 mb-4 flex items-center gap-2">
          <Target className="h-5 w-5 text-violet-500" /> Model Details
        </h2>
        <p className="text-sm text-zinc-500 mb-4">Click on any model to learn how it works</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {sortedModels.map((model) => (
            <ModelCard key={model.name} model={model} isBest={model.name === data.best_model} />
          ))}
        </div>
      </div>

      {/* Metric Explainer */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-lg text-zinc-100">Understanding the Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-1">
              <p className="text-violet-400 font-semibold">Accuracy</p>
              <p className="text-zinc-400">How many emails did it classify correctly overall?</p>
            </div>
            <div className="space-y-1">
              <p className="text-violet-400 font-semibold">Precision</p>
              <p className="text-zinc-400">When it says "spam", how often is it actually spam?</p>
            </div>
            <div className="space-y-1">
              <p className="text-violet-400 font-semibold">Recall</p>
              <p className="text-zinc-400">Of all real spam emails, how many did it catch?</p>
            </div>
            <div className="space-y-1">
              <p className="text-violet-400 font-semibold">F1 Score</p>
              <p className="text-zinc-400">The balance between precision and recall — the single best metric.</p>
            </div>
            <div className="space-y-1">
              <p className="text-violet-400 font-semibold">AUC</p>
              <p className="text-zinc-400">How well can the model separate spam from ham across all thresholds? Closer to 100% = better.</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
