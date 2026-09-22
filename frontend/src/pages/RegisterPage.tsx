import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { AlertCircle, ArrowRight } from 'lucide-react';

export const RegisterPage: React.FC = () => {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [department, setDepartment] = useState('Computer Science and Engineering');
  const [designation, setDesignation] = useState('Assistant Professor');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);

  const { register, isLoading, error: authError, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  React.useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    const cleanName = fullName.trim();
    if (!cleanName) {
      setValidationError('Please enter your full name.');
      return;
    }

    const cleanEmail = email.trim().toLowerCase();
    if (!cleanEmail) {
      setValidationError('Please enter your faculty email address.');
      return;
    }

    if (!cleanEmail.endsWith('@klu.ac.in')) {
      setValidationError('Registration is restricted to @klu.ac.in emails.');
      return;
    }

    if (password.length < 8) {
      setValidationError('Password must be at least 8 characters long.');
      return;
    }

    if (password !== confirmPassword) {
      setValidationError('Passwords do not match.');
      return;
    }

    try {
      await register({
        full_name: cleanName,
        email: cleanEmail,
        department,
        designation,
        password,
      });
      navigate('/dashboard', { replace: true });
    } catch (err: any) {
      // Error message is sanitized and set in AuthContext
    }
  };

  const displayError = validationError || (typeof authError === 'string' ? authError : null);

  return (
    <div className="flex min-h-screen items-center justify-center p-4 sm:p-6 bg-slate-950">
      <div className="w-full max-w-lg space-y-6">
        <div className="text-center">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white font-heading">
            Faculty Registration
          </h1>
        </div>

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
                Full Name
              </label>
              <input
                type="text"
                required
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Dr. Ramesh Kumar"
                className="input-field"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                Email (@klu.ac.in)
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name.department@klu.ac.in"
                className="input-field"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                  Department
                </label>
                <select
                  value={department}
                  onChange={(e) => setDepartment(e.target.value)}
                  className="input-field appearance-none bg-slate-900 cursor-pointer"
                >
                  <option value="Computer Science and Engineering">Computer Science (CSE)</option>
                  <option value="Electronics and Communication">Electronics (ECE)</option>
                  <option value="Electrical and Electronics">Electrical (EEE)</option>
                  <option value="Mechanical Engineering">Mechanical Engg</option>
                  <option value="Biotechnology">Biotechnology</option>
                  <option value="Mathematics & Sciences">Mathematics & Sciences</option>
                  <option value="Management Studies">Management Studies</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                  Designation
                </label>
                <select
                  value={designation}
                  onChange={(e) => setDesignation(e.target.value)}
                  className="input-field appearance-none bg-slate-900 cursor-pointer"
                >
                  <option value="Assistant Professor">Assistant Professor</option>
                  <option value="Associate Professor">Associate Professor</option>
                  <option value="Professor">Professor</option>
                  <option value="Department Head">Department Head</option>
                  <option value="Dean">Dean</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                  Password
                </label>
                <input
                  type="password"
                  required
                  minLength={8}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Min 8 characters"
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-1.5">
                  Confirm Password
                </label>
                <input
                  type="password"
                  required
                  minLength={8}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter password"
                  className="input-field"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full py-3.5 text-sm font-semibold mt-2"
            >
              <span>{isLoading ? 'Creating Faculty Account...' : 'Complete Registration'}</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </form>

          <div className="mt-6 pt-4 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
            <span>Already have an account?</span>
            <Link
              to="/login"
              className="font-semibold text-indigo-400 hover:text-indigo-300 transition-colors"
            >
              Sign In
            </Link>
          </div>
        </div>

        <div className="text-center">
          <Link to="/" className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
            ← Back
          </Link>
        </div>
      </div>
    </div>
  );
};
