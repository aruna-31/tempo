import React from 'react';
import { Navigate } from 'react-router-dom';

/**
 * Exam Mode has been removed in favor of HOD Academic & Pedagogical Monitoring.
 * Redirecting any legacy references to /admin-monitoring.
 */
export const ExamDashboardPage: React.FC = () => {
  return <Navigate to="/admin-monitoring" replace />;
};

export default ExamDashboardPage;
