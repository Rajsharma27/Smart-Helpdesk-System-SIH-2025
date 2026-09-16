"use client";

import React, { useState, useMemo, useEffect, use } from "react";
import { useRouter } from "next/navigation";
import { useUser } from "@/context/UserContext";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle,
  Search,
  Ticket,
  RefreshCcw,
} from "lucide-react";
import TicketStatics from "../card/TicketStatics";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "../select";
import { Separator } from "../separator";
import { SearchX } from "lucide-react";
import TicketCard from "../card/TicketCard";
import { fetchAllMyTickets } from "@/services/apiTicket";
import { BounceLoader } from "react-spinners";
import { Tabs, TabsList, TabsTrigger } from "../tabs";
import RaisedTicketCard from "../card/RaisedTicketCard";

const MyTickets = () => {
  const { currentUser } = useUser();

  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [priorityFilter, setPriorityFilter] = useState("all");
  const [assignedTickets, setAssignedTickets] = useState([]);
  const [raisedTickets, setRaisedTickets] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [ticketTab, setTicketTab] = useState("my-tickets");

  const fetchMyTickets = async () => {
    console.log("Fetching my tickets...");
    try {
      const result = await fetchAllMyTickets();

      console.log("Fetched tickets:", result);

      setAssignedTickets(result?.data?.assignedTo || []);
    } catch (error) {
      console.error(error.message);
      setAssignedTickets([]);
    }
  };

  const fetchRaisedTickets = async () => {
    console.log("Fetching raised tickets...");
    try {
      const result = await fetchAllMyTickets();

      console.log("Fetched tickets:", result);

      setRaisedTickets(result?.data?.createdBy || []);
    } catch (error) {
      console.error(error.message);
      setRaisedTickets([]);
    }
  };

  useEffect(() => {
    if (!currentUser) return;

    const fetchData = async () => {
      setIsLoading(true);
      await fetchRaisedTickets();
      await fetchMyTickets();
      setIsLoading(false);
    };

    fetchData();
  }, [currentUser]);

  // Calculate statistics
  const statistics = useMemo(() => {
    const total = assignedTickets.length;
    const open = assignedTickets.filter(
      (ticket) => ticket.status === "Open"
    ).length;
    const inProgress = assignedTickets.filter(
      (ticket) => ticket.status === "In Progress"
    ).length;
    const resolved = assignedTickets.filter(
      (ticket) => ticket.status === "Resolved"
    ).length;
    const closed = assignedTickets.filter(
      (ticket) => ticket.status === "Closed"
    ).length;
    const active = open + inProgress;

    return { total, open, inProgress, resolved, closed, active };
  }, [assignedTickets]);

  const raisedStatistics = useMemo(() => {
    const total = raisedTickets.length;
    const open = raisedTickets.filter(
      (ticket) => ticket.status === "Open"
    ).length;
    const inProgress = raisedTickets.filter(
      (ticket) => ticket.status === "In Progress"
    ).length;
    const resolved = raisedTickets.filter(
      (ticket) => ticket.status === "Resolved"
    ).length;
    const closed = raisedTickets.filter(
      (ticket) => ticket.status === "Closed"
    ).length;
    const active = open + inProgress;

    return { total, open, inProgress, resolved, closed, active };
  }, [raisedTickets]);

  // Filter tickets based on search and filters
  const filteredAssignedTickets = useMemo(() => {
    return assignedTickets.filter((ticket) => {
      const matchesSearch =
        ticket.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        ticket.description.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesStatus =
        statusFilter === "all" || ticket.status === statusFilter;
      const matchesPriority =
        priorityFilter === "all" || ticket.priority === priorityFilter;

      return matchesSearch && matchesStatus && matchesPriority;
    });
  }, [assignedTickets, searchTerm, statusFilter, priorityFilter]);

  const filteredRaisedTickets = useMemo(() => {
    return raisedTickets.filter((ticket) => {
      const matchesSearch =
        ticket.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        ticket.description.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesStatus =
        statusFilter === "all" || ticket.status === statusFilter;
      const matchesPriority =
        priorityFilter === "all" || ticket.priority === priorityFilter;

      return matchesSearch && matchesStatus && matchesPriority;
    });
  }, [raisedTickets, searchTerm, statusFilter, priorityFilter]);

  if (!currentUser || isLoading) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <BounceLoader color="#4A90E2" />
      </div>
    );
  }

  const resetFilters = () => {
    setSearchTerm("");
    setStatusFilter("all");
    setPriorityFilter("all");
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="space-y-1">
          <h1 className="text-3xl font-semibold text-primary">
            Hello {currentUser.user.name},
          </h1>
          <p className="text-muted-foreground text-lg">
            Here is what is happening with your tickets
          </p>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-2 gap-6 2xl:grid-cols-4">
        <TicketStatics
          label={"Total tickets"}
          value={
            ticketTab === "my-tickets"
              ? statistics.total
              : raisedStatistics.total
          }
          icon={Ticket}
        />

        <TicketStatics
          label={"Active tickets"}
          value={
            ticketTab === "my-tickets"
              ? statistics.active
              : raisedStatistics.active
          }
          icon={RefreshCcw}
          color={"blue"}
        />

        <TicketStatics
          label={"In progress"}
          value={
            ticketTab === "my-tickets"
              ? statistics.inProgress
              : raisedStatistics.inProgress
          }
          icon={Clock}
          color={"yellow"}
        />

        <TicketStatics
          label={"Resolved"}
          value={
            ticketTab === "my-tickets"
              ? statistics.resolved
              : raisedStatistics.resolved
          }
          icon={CheckCircle}
          color={"green"}
        />
      </div>

      {/* Filters and Search */}
      <Card>
        <CardContent>
          <div className="grid grid-cols-4 grid-rows-2 gap-x-5 items-center">
            <Label htmlFor="search" className={"text-lg"}>
              Search
            </Label>
            <Label htmlFor="status" className={"text-lg"}>
              Status
            </Label>
            <Label htmlFor="priority" className={"text-lg"}>
              Priority
            </Label>
            <div></div>

            <div className="relative">
              <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                id="search"
                placeholder="Search tickets..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8"
              />
            </div>

            <Select
              value={statusFilter}
              onValueChange={(value) => setStatusFilter(value)}
            >
              <SelectTrigger className={"w-full"}>
                <SelectValue placeholder="All Statuses" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel>Statuses</SelectLabel>
                  <SelectItem value="all">All Statuses</SelectItem>
                  <SelectItem value="Open">Open</SelectItem>
                  <SelectItem value="In progress">In Progress</SelectItem>
                  <SelectItem value="Resolved">Resolved</SelectItem>
                  <SelectItem value="Closed">Closed</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>

            <Select
              value={priorityFilter}
              onValueChange={(value) => setPriorityFilter(value)}
            >
              <SelectTrigger className={"w-full"}>
                <SelectValue placeholder="All Priorities" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel>Priorities</SelectLabel>
                  <SelectItem value="all">All Priorities</SelectItem>
                  <SelectItem value="High">High</SelectItem>
                  <SelectItem value="Medium">Medium</SelectItem>
                  <SelectItem value="Low">Low</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>

            <Button className={"flex-1 h-full"} onClick={resetFilters}>
              Reset filters
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Tickets List */}
      <div className="space-y-4 mt-12">
        <div className="space-x-2">
          <span className="text-3xl font-semibold text-chart-3">
            Tickets (
            {ticketTab === "my-tickets"
              ? filteredAssignedTickets.length
              : filteredRaisedTickets.length}
            )
          </span>
          <Separator className={"mt-2"} />
        </div>

        <div>
          <Tabs defaultValue="my-tickets" onValueChange={setTicketTab}>
            <TabsList>
              <TabsTrigger value="my-tickets">Assigned to me</TabsTrigger>
              <TabsTrigger value="raised-tickets">Raised by me</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        {/* Showing the tickets depending on the selected tab */}
        {ticketTab === "my-tickets" ? (
          <>
            {filteredAssignedTickets.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12">
                <SearchX className="h-16 w-16 text-muted-foreground mb-4 opacity-70" />
                <h3 className="text-2xl text-muted-foreground font-semibold mb-2">
                  No tickets found
                </h3>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-6">
                {filteredAssignedTickets.map((ticket) => (
                  <TicketCard key={ticket.title} ticket={ticket} />
                ))}
              </div>
            )}
          </>
        ) : (
          <>
            {filteredRaisedTickets.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12">
                <SearchX className="h-16 w-16 text-muted-foreground mb-4 opacity-70" />
                <h3 className="text-2xl text-muted-foreground font-semibold mb-2">
                  No tickets found
                </h3>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-6">
                {filteredRaisedTickets.map((ticket) => (
                  <RaisedTicketCard key={ticket.title} ticket={ticket} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default MyTickets;
