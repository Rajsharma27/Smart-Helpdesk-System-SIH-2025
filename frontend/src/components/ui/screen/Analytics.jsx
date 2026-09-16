"use client";

import React, { useState, useMemo } from "react";
import { useUser } from "@/context/UserContext";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  BarChart3,
  PieChart as PieChartIcon,
  TrendingUp,
  TrendingDown,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  Users,
  Calendar,
  Filter,
  Download,
  RefreshCw,
  Activity,
  Target,
  Zap,
  AlertTriangle,
} from "lucide-react";

// Mock analytics data
const analyticsData = {
  overview: {
    totalTickets: 1247,
    openTickets: 89,
    resolvedTickets: 1089,
    avgResolutionTime: 4.2, // hours
    customerSatisfaction: 4.6,
    trendsLastMonth: {
      totalTickets: +12.5,
      resolutionTime: -8.3,
      satisfaction: +2.1,
    },
  },
  ticketsByStatus: [
    { status: "Open", count: 89, color: "#3b82f6", percentage: 7.1 },
    { status: "In Progress", count: 134, color: "#f59e0b", percentage: 10.7 },
    { status: "Resolved", count: 956, color: "#10b981", percentage: 76.7 },
    { status: "Closed", count: 68, color: "#6b7280", percentage: 5.5 },
  ],
  ticketsByCategory: [
    { category: "Network", count: 324, color: "#ef4444" },
    { category: "Hardware", count: 267, color: "#f97316" },
    { category: "Software", count: 198, color: "#eab308" },
    { category: "Email", count: 156, color: "#22c55e" },
    { category: "Database", count: 134, color: "#06b6d4" },
    { category: "Security", count: 98, color: "#8b5cf6" },
    { category: "Access", count: 70, color: "#ec4899" },
  ],
  ticketsByPriority: [
    { priority: "Critical", count: 23, color: "#dc2626" },
    { priority: "High", count: 156, color: "#ea580c" },
    { priority: "Medium", count: 687, color: "#ca8a04" },
    { priority: "Low", count: 381, color: "#16a34a" },
  ],
  monthlyTrends: [
    { month: "Jan", created: 98, resolved: 89 },
    { month: "Feb", created: 112, resolved: 105 },
    { month: "Mar", created: 134, resolved: 128 },
    { month: "Apr", created: 156, resolved: 149 },
    { month: "May", created: 143, resolved: 138 },
    { month: "Jun", created: 167, resolved: 162 },
    { month: "Jul", created: 189, resolved: 184 },
    { month: "Aug", created: 178, resolved: 173 },
    { month: "Sep", created: 165, resolved: 161 },
    { month: "Oct", created: 123, resolved: 118 },
  ],
  teamPerformance: [
    { team: "Network Team", resolved: 234, avgTime: 3.2, satisfaction: 4.7 },
    {
      team: "Hardware Support",
      resolved: 189,
      avgTime: 2.8,
      satisfaction: 4.5,
    },
    { team: "Software Team", resolved: 167, avgTime: 5.1, satisfaction: 4.3 },
    { team: "Database Team", resolved: 145, avgTime: 6.2, satisfaction: 4.8 },
    { team: "Security Team", resolved: 98, avgTime: 4.5, satisfaction: 4.6 },
  ],
  resolutionTimes: [
    { range: "< 1 hour", count: 234, percentage: 18.8 },
    { range: "1-4 hours", count: 456, percentage: 36.6 },
    { range: "4-24 hours", count: 378, percentage: 30.3 },
    { range: "1-3 days", count: 134, percentage: 10.7 },
    { range: "> 3 days", count: 45, percentage: 3.6 },
  ],
};

// Simple bar chart component
const BarChart = ({ data, height = 200, title }) => {
  const maxValue = Math.max(
    ...data.map((item) => item.count || item.resolved || 0)
  );

  return (
    <div className="space-y-3">
      {title && <h4 className="font-medium text-sm">{title}</h4>}
      <div className="flex items-end space-x-2" style={{ height }}>
        {data.map((item, index) => {
          const value = item.count || item.resolved || 0;
          const heightPercentage = (value / maxValue) * 100;
          const color = item.color || "#3b82f6";

          return (
            <div key={index} className="flex flex-col items-center flex-1">
              <div className="relative w-full">
                <div
                  className="w-full rounded-t-md transition-all duration-300 hover:opacity-80"
                  style={{
                    height: `${heightPercentage}%`,
                    backgroundColor: color,
                    minHeight: "4px",
                  }}
                ></div>
              </div>
              <div className="text-xs text-center mt-2 font-medium">
                {value}
              </div>
              <div className="text-xs text-muted-foreground text-center">
                {item.category || item.team || item.month || item.priority}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Simple pie chart component
const CustomPieChart = ({ data, size = 200 }) => {
  const total = data.reduce((sum, item) => sum + item.count, 0);
  let currentAngle = 0;

  const radius = size / 2 - 10;
  const centerX = size / 2;
  const centerY = size / 2;

  return (
    <div className="relative">
      <svg width={size} height={size} className="transform -rotate-90">
        {data.map((item, index) => {
          const percentage = item.count / total;
          const angle = percentage * 360;
          const x1 =
            centerX + radius * Math.cos((currentAngle * Math.PI) / 180);
          const y1 =
            centerY + radius * Math.sin((currentAngle * Math.PI) / 180);
          const x2 =
            centerX +
            radius * Math.cos(((currentAngle + angle) * Math.PI) / 180);
          const y2 =
            centerY +
            radius * Math.sin(((currentAngle + angle) * Math.PI) / 180);

          const largeArcFlag = angle > 180 ? 1 : 0;

          const pathData = [
            `M ${centerX} ${centerY}`,
            `L ${x1} ${y1}`,
            `A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2}`,
            "Z",
          ].join(" ");

          currentAngle += angle;

          return (
            <path
              key={index}
              d={pathData}
              fill={item.color}
              stroke="white"
              strokeWidth="2"
              className="hover:opacity-80 transition-opacity"
            />
          );
        })}
      </svg>

      {/* Legend */}
      <div className="absolute top-0 right-0 space-y-1 transform translate-x-full ml-4">
        {data.map((item, index) => (
          <div key={index} className="flex items-center space-x-2 text-xs">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: item.color }}
            ></div>
            <span>{item.status || item.priority}</span>
            <span className="text-muted-foreground">({item.count})</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Line chart component
const LineChart = ({ data, height = 200 }) => {
  const maxValue = Math.max(
    ...data.flatMap((item) => [item.created, item.resolved])
  );
  const width = 400;
  const padding = 40;

  const points = data.map((item, index) => {
    const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
    const createdY =
      height - padding - (item.created / maxValue) * (height - 2 * padding);
    const resolvedY =
      height - padding - (item.resolved / maxValue) * (height - 2 * padding);
    return { x, createdY, resolvedY, month: item.month };
  });

  const createdPath = points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${point.x} ${point.createdY}`
    )
    .join(" ");

  const resolvedPath = points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${point.x} ${point.resolvedY}`
    )
    .join(" ");

  return (
    <div className="relative">
      <svg width={width} height={height}>
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((ratio, index) => {
          const y = padding + ratio * (height - 2 * padding);
          return (
            <line
              key={index}
              x1={padding}
              y1={y}
              x2={width - padding}
              y2={y}
              stroke="#e5e7eb"
              strokeWidth="1"
            />
          );
        })}

        {/* Created line */}
        <path
          d={createdPath}
          fill="none"
          stroke="#ef4444"
          strokeWidth="2"
          className="drop-shadow-sm"
        />

        {/* Resolved line */}
        <path
          d={resolvedPath}
          fill="none"
          stroke="#22c55e"
          strokeWidth="2"
          className="drop-shadow-sm"
        />

        {/* Data points */}
        {points.map((point, index) => (
          <g key={index}>
            <circle cx={point.x} cy={point.createdY} r="4" fill="#ef4444" />
            <circle cx={point.x} cy={point.resolvedY} r="4" fill="#22c55e" />
            <text
              x={point.x}
              y={height - 10}
              textAnchor="middle"
              fontSize="10"
              fill="#6b7280"
            >
              {point.month}
            </text>
          </g>
        ))}
      </svg>

      {/* Legend */}
      <div className="flex items-center space-x-4 mt-4 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <span>Created</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span>Resolved</span>
        </div>
      </div>
    </div>
  );
};

const Analytics = () => {
  const { currentUser } = useUser();
  const [timeRange, setTimeRange] = useState("month");
  const [selectedCategory, setSelectedCategory] = useState("all");

  // Don't render if no user
  if (!currentUser) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Clock className="h-8 w-8 animate-spin mx-auto mb-4" />
            <p>Loading...</p>
          </div>
        </div>
      </div>
    );
  }

  const {
    overview,
    ticketsByStatus,
    ticketsByCategory,
    ticketsByPriority,
    monthlyTrends,
    teamPerformance,
    resolutionTimes,
  } = analyticsData;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Analytics Dashboard
          </h1>
          <p className="text-muted-foreground">
            Comprehensive insights into IT support performance
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Filter className="h-4 w-4 mr-2" />
            Filter
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Overview KPIs */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tickets</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {overview.totalTickets.toLocaleString()}
            </div>
            <div className="flex items-center text-xs text-green-600">
              <TrendingUp className="h-3 w-3 mr-1" />+
              {overview.trendsLastMonth.totalTickets}% from last month
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Open Tickets</CardTitle>
            <AlertCircle className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">
              {overview.openTickets}
            </div>
            <p className="text-xs text-muted-foreground">Awaiting resolution</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Avg Resolution Time
            </CardTitle>
            <Clock className="h-4 w-4 text-orange-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">
              {overview.avgResolutionTime}h
            </div>
            <div className="flex items-center text-xs text-green-600">
              <TrendingDown className="h-3 w-3 mr-1" />-
              {overview.trendsLastMonth.resolutionTime}% improvement
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Customer Satisfaction
            </CardTitle>
            <Target className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {overview.customerSatisfaction}/5
            </div>
            <div className="flex items-center text-xs text-green-600">
              <TrendingUp className="h-3 w-3 mr-1" />+
              {overview.trendsLastMonth.satisfaction}% increase
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Resolution Rate
            </CardTitle>
            <CheckCircle className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {(
                (overview.resolvedTickets / overview.totalTickets) *
                100
              ).toFixed(1)}
              %
            </div>
            <p className="text-xs text-muted-foreground">
              {overview.resolvedTickets} resolved
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 1 */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Ticket Status Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <PieChartIcon className="h-5 w-5" />
              Tickets by Status
            </CardTitle>
            <CardDescription>
              Current distribution of ticket statuses
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center">
              <CustomPieChart data={ticketsByStatus} size={220} />
            </div>
          </CardContent>
        </Card>

        {/* Priority Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              Tickets by Priority
            </CardTitle>
            <CardDescription>Priority level distribution</CardDescription>
          </CardHeader>
          <CardContent>
            <BarChart data={ticketsByPriority} height={200} />
            <div className="grid grid-cols-2 gap-4 mt-4">
              {ticketsByPriority.map((item, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between text-sm"
                >
                  <div className="flex items-center space-x-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    ></div>
                    <span>{item.priority}</span>
                  </div>
                  <span className="font-medium">{item.count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Category Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Tickets by Category
            </CardTitle>
            <CardDescription>Issue categories breakdown</CardDescription>
          </CardHeader>
          <CardContent>
            <BarChart data={ticketsByCategory} height={220} />
          </CardContent>
        </Card>

        {/* Monthly Trends */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Monthly Trends
            </CardTitle>
            <CardDescription>
              Tickets created vs resolved over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <LineChart data={monthlyTrends} height={220} />
          </CardContent>
        </Card>
      </div>

      {/* Team Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Team Performance
          </CardTitle>
          <CardDescription>
            Individual team metrics and performance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Team</th>
                  <th className="text-right p-2">Tickets Resolved</th>
                  <th className="text-right p-2">Avg Resolution Time</th>
                  <th className="text-right p-2">Satisfaction Score</th>
                  <th className="text-right p-2">Performance</th>
                </tr>
              </thead>
              <tbody>
                {teamPerformance.map((team, index) => (
                  <tr key={index} className="border-b hover:bg-muted/50">
                    <td className="p-2 font-medium">{team.team}</td>
                    <td className="p-2 text-right">{team.resolved}</td>
                    <td className="p-2 text-right">{team.avgTime}h</td>
                    <td className="p-2 text-right">{team.satisfaction}/5</td>
                    <td className="p-2 text-right">
                      <Badge
                        className={
                          team.satisfaction >= 4.5
                            ? "bg-green-100 text-green-800"
                            : team.satisfaction >= 4.0
                            ? "bg-yellow-100 text-yellow-800"
                            : "bg-red-100 text-red-800"
                        }
                      >
                        {team.satisfaction >= 4.5
                          ? "Excellent"
                          : team.satisfaction >= 4.0
                          ? "Good"
                          : "Needs Improvement"}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Resolution Time Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Resolution Time Analysis
          </CardTitle>
          <CardDescription>
            Distribution of ticket resolution times
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {resolutionTimes.map((item, index) => (
              <div key={index} className="flex items-center space-x-4">
                <div className="w-24 text-sm font-medium">{item.range}</div>
                <div className="flex-1">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${item.percentage}%` }}
                    ></div>
                  </div>
                </div>
                <div className="w-16 text-sm text-right">{item.count}</div>
                <div className="w-16 text-sm text-muted-foreground text-right">
                  {item.percentage}%
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Analytics;
