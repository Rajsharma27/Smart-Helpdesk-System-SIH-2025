"use client";

import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import {
  Bot,
  User,
  AlertCircle,
  HelpCircle,
  Settings,
  RefreshCw,
} from "lucide-react";
import PrompBox from "../input/PrompBox";
import axios from "axios";
import { createTicket } from "@/services/apiTicket";

// Predefined quick actions
const quickActions = [
  { label: "Create New Ticket", value: "Create new ticket", icon: AlertCircle },
  {
    label: "Network Issues",
    value: "I am experiencing network issues",
    icon: Settings,
  },
  {
    label: "Password Reset",
    value: "I need help with password reset",
    icon: RefreshCw,
  },
  { label: "Need Help", value: "I need help", icon: HelpCircle },
];

const API_BASE_URL = process.env.NEXT_PUBLIC_PYTHON_SERVER_URL || "http://localhost:8000";
const sessionId = "user123"; // This should be dynamically set based on the logged-in user

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "bot",
      content:
        "Hi! I'm PowerGrid's IT Support Assistant. I'm here to help you with your IT-related questions and issues. How can I assist you today?",
      timestamp: new Date(),
      status: "delivered",
    },
  ]);
  const [message, setMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const formatBotMessage = (data) => {
    let message = "";

    if (data.responseText) {
      message += data.responseText + "\n\n";
    }

    if (Array.isArray(data.solution) && data.solution.length > 0) {
      message += "Here are some suggested solutions:\n";
      data.solution.forEach((step) => {
        message += `\n${step}`;
      });
    }

    return message.trim();
  };

  // Send message function
  const sendMessage = async () => {
    setMessage("");
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        type: "user",
        content: message,
        timestamp: new Date(),
        status: "sent",
      },
    ]);

    setIsTyping(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: message,
        session_id: sessionId,
      });

      console.log("Chatbot response:", response.data);
      // title, description, priority, category, subcategory
      if (response.data.ticket) {
        const { title, description, priority, category, subcategory, tags } =
          response.data.ticket;
        // Create ticket in the backend
        try {
          await createTicket({
            title,
            description,
            priority,
            category,
            subcategory,
            tags,
          });

          setMessages((prev) => [
            ...prev,
            {
              id: Date.now(),
              type: "bot",
              content: `✅ **Ticket Created Successfully!**\n\n**Title:** ${title}\n**Priority:** ${priority}\n**Category:** ${category}\n\nYour ticket has been submitted and assigned to a support agent. You can track its progress in the "My Tickets" section.`,
              timestamp: new Date(),
              status: "received",
            },
          ]);

          return;
        } catch (error) {
          console.error("Error creating ticket:", error);
        }
      }

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          type: "bot",
          content: formatBotMessage(response.data),
          timestamp: new Date(),
          status: "received",
        },
      ]);
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          type: "bot",
          content:
            "Sorry, I couldn't fetch a response. Please try again later.",
          timestamp: new Date(),
          status: "error",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  // Handle quick action clicks
  const handleQuickAction = async (action) => {
    setMessage("");
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        type: "user",
        content: action,
        timestamp: new Date(),
        status: "sent",
      },
    ]);

    setIsTyping(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: action, // 👈 use the quick action text
        session_id: sessionId,
      });

      console.log("Chatbot response:", response.data);
      // title, description, priority, category, subcategory
      if (response.data.ticket) {
        const { title, description, priority, category, subcategory, tags } =
          response.data.ticket;
        // Create ticket in the backend
        try {
          await createTicket({
            title,
            description,
            priority,
            category,
            subcategory,
            tags,
          });

          setMessages((prev) => [
            ...prev,
            {
              id: Date.now(),
              type: "bot",
              content: `I have raised a ticket for regarding your issue. You can check the ticket in the "My Tickets" section.`,
              timestamp: new Date(),
              status: "received",
            },
          ]);

          return;
        } catch (error) {
          console.error("Error creating ticket:", error);
        }
      }

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          type: "bot",
          content: formatBotMessage(response.data),
          timestamp: new Date(),
          status: "received",
        },
      ]);
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          type: "bot",
          content:
            "Sorry, I couldn't fetch a response. Please try again later.",
          timestamp: new Date(),
          status: "error",
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="flex flex-col justify-between overflow-hidden h-[calc(100vh-2rem)] max-h-[850px]">
      {/* Messages Area */}
      <div
        className="flex-1 overflow-y-auto p-4 space-y-4"
        style={{
          scrollbarWidth: "thin",
          scrollbarColor: "muted-foreground transparent",
          scrollbarGutter: "stable",
        }}
      >
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.type === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`flex max-w-[80%] gap-2 ${
                message.type === "bot" ? "flex-row" : "flex-row-reverse"
              } items-start space-x-2`}
            >
              {/* Avatar */}
              {message.type === "user" ? (
                <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center">
                  <User className="h-6 w-6 text-primary-foreground" />
                </div>
              ) : (
                <div className="w-10 h-10 rounded-full bg-primary-foreground flex items-center justify-center">
                  <Bot className="h-6 w-6 text-primary" />
                </div>
              )}

              {/* Message Bubble */}
              <div className="flex flex-col">
                <div
                  className={`rounded-lg px-3 py-2 ${
                    message.type === "bot"
                      ? "bg-chart-1 text-primary-foreground"
                      : "bg-primary text-primary-foreground"
                  }`}
                >
                  <div className="prose prose-sm max-w-none text-base text-primary-foreground">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  </div>
                </div>
                <div
                  className={`flex items-center text-xs text-muted-foreground mt-1 ${
                    message.type === "bot" ? "justify-start" : "justify-end"
                  }`}
                >
                  {formatTime(message.timestamp)}
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="flex items-start space-x-2">
              <div className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
                <Bot className="h-4 w-4 text-blue-600" />
              </div>
              <div className="bg-gray-100 rounded-lg px-3 py-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.1s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      <div className="p-4 space-y-4">
        <div className="flex flex-wrap gap-2">
          {quickActions.map((action, index) => {
            const IconComponent = action.icon;
            return (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => handleQuickAction(action.value)}
                className="text-xs cursor-pointer"
              >
                <IconComponent className="h-3 w-3 mr-1" />
                {action.label}
              </Button>
            );
          })}
        </div>

        {/* Message Input */}
        <PrompBox
          onSend={sendMessage}
          message={message}
          setMessage={setMessage}
        />
      </div>
    </div>
  );
};

export default Chatbot;
