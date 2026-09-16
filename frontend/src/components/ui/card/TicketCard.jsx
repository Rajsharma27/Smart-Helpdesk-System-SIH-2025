import { Card, CardContent, CardFooter, CardHeader } from "../card";
import { Tag } from "lucide-react";
import {
  Select,
  SelectItem,
  SelectTrigger,
  SelectContent,
  SelectValue,
} from "../select";
import { updateTicket } from "@/services/apiTicket";
import { useState } from "react";
import { Badge } from "../badge";

const TicketCard = ({ ticket }) => {
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };
  const [ticketStatus, setTicketStatus] = useState(ticket.status);

  const handleStatusChange = async (newStatus) => {
    // Implement status change logic here, e.g., make an API call to update the ticket status
    try {
      setTicketStatus(newStatus);
      await updateTicket(ticket._id, { status: newStatus });
    } catch (error) {
      console.error("Error updating ticket status:", error);
      setTicketStatus(ticket.status); // Revert to previous status on error
    }
  };

  const badgeVariant = {
    Open: "green",
    "In Progress": "yellow",
    Closed: "red",
    Low: "green",
    Medium: "yellow",
    High: "red",
  };

  return (
    <Card>
      <CardContent className={"flex flex-col gap-4 h-full"}>
        <div className={"flex justify-between"}>
          <span className="text-muted-foreground">{ticket._id}</span>
          <span>{formatDate(ticket.createdAt)}</span>
        </div>

        <div className="flex-1 space-y-1">
          <h2 className="text-2xl font-semibold text-primary">
            {ticket.title}
          </h2>
          <p className="text-base font-medium text-muted-foreground leading-6">
            {ticket.description}
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2 text-muted-foreground">
              <Tag size={16} />
              <span className="text-sm">{ticket.tags.join(", ")}Hardware</span>
            </div>
          </div>

          <div className="flex justify-between items-center">
            <Badge variant={badgeVariant[ticket.priority] || "default"}>
              <span className="uppercase">{ticket.priority}</span>
            </Badge>
            <Select value={ticketStatus} onValueChange={handleStatusChange}>
              <SelectTrigger variant={badgeVariant[ticketStatus]}>
                <SelectValue placeholder={ticketStatus} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Open">Open</SelectItem>
                <SelectItem value="In Progress">In Progress</SelectItem>
                <SelectItem value="Closed">Closed</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TicketCard;
