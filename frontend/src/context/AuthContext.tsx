import React, { createContext, useContext, useEffect, useState } from 'react';
import { api } from '../api/client';
import { Faculty } from '../types';

interface AuthContextType {
  user: Faculty | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (data: {
    email: string;
    password: string;
    full_name: string;
    department: string;
    designation?: string;
  }) => Promise<void>;
  logout: () => void;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<Faculty | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('tempo_access_token'));
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem('tempo_access_token');
      if (storedToken) {
        try {
          const faculty = await api.getMe();
          setUser(faculty);
          setToken(storedToken);
        } catch (err) {
          console.warn('Session expired or invalid token:', err);
          logout();
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    setError(null);
    setIsLoading(true);

    const cleanEmail = email.trim().toLowerCase();
    if (!cleanEmail.endsWith('@klu.ac.in')) {
      setIsLoading(false);
      const errMsg = 'Access restricted: Only authorized KLU faculty accounts (@klu.ac.in) are permitted.';
      setError(errMsg);
      throw new Error(errMsg);
    }

    try {
      const res = await api.login({ email: cleanEmail, password });
      setToken(res.access_token);
      const faculty = await api.getMe();
      setUser(faculty);
    } catch (err: any) {
      const msg = typeof err?.message === 'string' ? err.message : 'Login failed';
      setError(msg);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (data: {
    email: string;
    password: string;
    full_name: string;
    department: string;
    designation?: string;
  }) => {
    setError(null);
    setIsLoading(true);

    const cleanEmail = data.email.trim().toLowerCase();
    if (!cleanEmail.endsWith('@klu.ac.in')) {
      setIsLoading(false);
      const errMsg = 'Access restricted: Registration is restricted to official university emails ending in @klu.ac.in';
      setError(errMsg);
      throw new Error(errMsg);
    }

    try {
      await api.register({
        ...data,
        email: cleanEmail,
      });
      // Auto-login after registration
      await login(cleanEmail, data.password);
    } catch (err: any) {
      const msg = typeof err?.message === 'string' ? err.message : 'Registration failed';
      setError(msg);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    api.clearTokens();
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token && !!user,
        isLoading,
        login,
        register,
        logout,
        error,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
