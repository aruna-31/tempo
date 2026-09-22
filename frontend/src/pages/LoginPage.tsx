import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { AlertCircle, ArrowRight, Lock, Mail } from 'lucide-react';

export const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);

  const { login, isLoading, error: authError, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    const cleanEmail = email.trim().toLowerCase();
    if (!cleanEmail) {
      setValidationError('Please enter your faculty email address.');
      return;
    }

    if (!cleanEmail.endsWith('@klu.ac.in')) {
      setValidationError('Access restricted: Please enter an official faculty email ending with @klu.ac.in');
      return;
    }

    if (!password) {
      setValidationError('Please enter your password.');
      return;
    }

    try {
      await login(cleanEmail, password);
      navigate('/dashboard', { replace: true });
    } catch (err: any) {
      // Error message is captured and sanitized in AuthContext
    }
  };

  const displayError = validationError || (typeof authError === 'string' ? authError : null);

  return (
    <div className="flex min-h-screen items-center justify-center p-4 sm:p-6 bg-slate-950">
      <div className="w-full max-w-md space-y-6">
        {/* Brand Header */}
        <div className="text-center space-y-3">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white font-heading">
              TEMPO
            </h1>
            <h2 className="text-lg font-semibold text-indigo-400 tracking-wide mt-0.5">
              Faculty Portal
            </h2>
          </div>
        </div>

        {/* Login Glass Panel */}
        <div className="glass-panel p-6 sm:p-8 shadow-2xl border border-slate-800 bg-slate-900/80 backdrop-blur-xl">
          {displayError && (
            <div className="mb-5 flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-300">
              <AlertCircle className="h-4 w-4 flex-shrink-0 text-rose-400 mt-0.5" />
              <span className="leading-relaxed">{displayError}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Faculty Email (@klu.ac.in)
              </label>
              <div className="relative">
                <Mail className="pointer-events-none absolute left-3.5 top-3.5 h-4 w-4 text-slate-500" />
                <input
                  type="email"
                  required
                  autoFocus
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="faculty.name@klu.ac.in"
                  className="input-field pl-10"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Password
              </label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3.5 top-3.5 h-4 w-4 text-slate-500" />
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="input-field pl-10"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full py-3.5 text-sm font-semibold mt-2"
            >
              <span>{isLoading ? 'Authenticating...' : 'Sign In to Portal'}</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </form>

          <div className="mt-6 pt-4 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
            <span>Don't have an account?</span>
            <Link
              to="/register"
              className="font-semibold text-indigo-400 hover:text-indigo-300 transition-colors"
            >
              Register Faculty Account →
            </Link>
          </div>
        </div>

        <div className="text-center">
          <Link
            to="/"
            className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
          >
            ← Back
          </Link>
        </div>
      </div>
    </div>
  );
};
