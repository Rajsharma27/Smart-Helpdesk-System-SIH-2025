import React from "react";
import { Card, CardContent } from "../card";
import { Tag } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "../badge";

const RaisedTicketCard = ({ ticket }) => {
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
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
      <CardContent className={"flex flex-col gap-4"}>
        <div className={"flex justify-between"}>
          <span className="text-muted-foreground">{ticket._id}</span>
          <span className="text-muted-foreground">
            {formatDate(ticket.createdAt)}
          </span>
        </div>

        <div className="space-y-1">
          <h2 className="text-2xl font-semibold text-primary">
            {ticket.title}
          </h2>
          <p className="text-base font-medium text-muted-foreground leading-6">
            {ticket.description}
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <span>
              {ticket.assignedTo ? ticket.assignedTo.name : "Unassigned"}
            </span>
            <div>
              <div className="flex items-center space-x-2 text-muted-foreground">
                <Tag />
                <span>{ticket.tags.join(", ")}</span>
              </div>
            </div>
          </div>

          <div className="flex justify-between items-center">
            <Badge variant={badgeVariant[ticket.priority] || "default"}>
              <span className="text-sm font-medium px-2">
                {ticket.priority}
              </span>
            </Badge>
            <div className={cn("border p-1 w-[100px] text-center rounded-sm")}>
              {ticket.status}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default RaisedTicketCard;
