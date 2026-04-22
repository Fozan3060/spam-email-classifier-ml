import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Mail, ShieldAlert, ShieldCheck, Loader2, LayoutDashboard, Send } from 'lucide-react'
import Dashboard from '@/components/Dashboard'
import { API_URL } from '@/lib/api'

type PredictionResult = {
  prediction: string
  is_spam: boolean
  confidence: number | null
  processed_text: string
}

function Classifier() {
  const [message, setMessage] = useState('')
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async () => {
    if (!message.trim()) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })

      if (!response.ok) throw new Error('API request failed')

      const data = await response.json()
      setResult(data)
    } catch {
      setError('Could not connect to the API. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-lg text-zinc-100">Email Message</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            placeholder="Type or paste an email message here..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            className="bg-zinc-950 border-zinc-700 text-zinc-100 placeholder:text-zinc-500 min-h-[160px]"
          />
          <Button
            onClick={handleSubmit}
            disabled={loading || !message.trim()}
            size="lg"
            className="w-full bg-violet-600 hover:bg-violet-700 text-white"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Classify Email'
            )}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Card className="bg-red-950/50 border-red-800">
          <CardContent className="pt-6">
            <p className="text-red-400 text-sm">{error}</p>
          </CardContent>
        </Card>
      )}

      {result && (
        <Card className={`border ${result.is_spam ? 'bg-red-950/30 border-red-800' : 'bg-green-950/30 border-green-800'}`}>
          <CardContent className="pt-6 space-y-4">
            <div className="flex items-center gap-3">
              {result.is_spam ? (
                <ShieldAlert className="h-10 w-10 text-red-500" />
              ) : (
                <ShieldCheck className="h-10 w-10 text-green-500" />
              )}
              <div>
                <h3 className={`text-2xl font-bold ${result.is_spam ? 'text-red-400' : 'text-green-400'}`}>
                  {result.is_spam ? 'Spam Detected' : 'Not Spam'}
                </h3>
                {result.confidence !== null && (
                  <p className="text-zinc-400 text-sm">
                    Confidence: {(result.confidence * 100).toFixed(1)}% spam probability
                  </p>
                )}
              </div>
            </div>
            <div className="pt-2 border-t border-zinc-800">
              <p className="text-zinc-500 text-xs mb-1">Processed text:</p>
              <p className="text-zinc-400 text-sm">{result.processed_text}</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function App() {
  const [tab, setTab] = useState<'classify' | 'dashboard'>('classify')

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 sticky top-0 bg-zinc-950/80 backdrop-blur-sm z-10">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mail className="h-6 w-6 text-violet-500" />
            <h1 className="text-xl font-bold">Spam Classifier</h1>
          </div>
          <div className="flex gap-2">
            <Button
              variant={tab === 'classify' ? 'default' : 'outline'}
              onClick={() => setTab('classify')}
              className={tab === 'classify' ? 'bg-violet-600 hover:bg-violet-700' : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100'}
            >
              <Send className="mr-2 h-4 w-4" /> Classify
            </Button>
            <Button
              variant={tab === 'dashboard' ? 'default' : 'outline'}
              onClick={() => setTab('dashboard')}
              className={tab === 'dashboard' ? 'bg-violet-600 hover:bg-violet-700' : 'border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100'}
            >
              <LayoutDashboard className="mr-2 h-4 w-4" /> Dashboard
            </Button>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-5xl mx-auto px-4 py-8">
        {tab === 'classify' ? (
          <div className="max-w-2xl mx-auto">
            <div className="text-center space-y-2 mb-6">
              <h2 className="text-2xl font-bold">Check if an email is spam</h2>
              <p className="text-zinc-400 text-sm">
                Paste an email message below and our ML model will classify it
              </p>
            </div>
            <Classifier />
          </div>
        ) : (
          <Dashboard />
        )}
      </main>
    </div>
  )
}

export default App
