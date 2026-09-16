"use client";

import { useState } from "react";
import { Separator } from "../separator";
import { Label } from "../label";
import { Input } from "../input";
import { Textarea } from "../textarea";
import { Button } from "../button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "../select";
import { createTicket } from "@/services/apiTicket";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";

const priorityLevels = [
  { label: "Low", value: "Low" },
  { label: "Medium", value: "Medium" },
  { label: "High", value: "High" },
];

const categories = [
  { label: "Password Reset", value: "Password Reset" },
  { label: "Hardware", value: "Hardware" },
  { label: "Software", value: "Software" },
  { label: "Network", value: "Network" },
  { label: "Other", value: "Other" },
];

const RaiseTicket = () => {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [formData, setFormData] = useState({
    title: "",
    description: "",
    priority: "Medium",
    category: "",
    subcategory: "",
    tags: [],
  });

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validation
    if (!formData.title.trim()) {
      toast("Title required", {
        description: "Title can't be empty",
      })
      return;
    }

    if (!formData.description.trim()) {
      toast.error("Validation Error", {
        description: "Please enter a ticket description",
      });
      return;
    }

    if (!formData.category) {
      toast.error("Validation Error", {
        description: "Please select a category",
      });
      return;
    }

    setIsSubmitting(true);

    try {
      const result = await createTicket({
        title: formData.title.trim(),
        description: formData.description.trim(),
        priority: formData.priority,
        category: formData.category,
        subcategory: formData.subcategory.trim(),
        tags: formData.tags.length > 0 ? formData.tags : [],
      });

      console.log("Ticket created:", result);

      toast.success("Success!", {
        description: "Your ticket has been created successfully",
      });

      // Reset form
      setFormData({
        title: "",
        description: "",
        priority: "Medium",
        category: "",
        subcategory: "",
        tags: [],
      });
    } catch (error) {
      console.error("Error creating ticket:", error);

      toast.error("Error", {
        description:
          error.message || "Failed to create ticket. Please try again.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-primary">Raise Ticket</h1>
        <p className="text-muted-foreground text-lg">
          You can raise an IT ticket here
        </p>
      </div>

      <Separator />

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Basic Details */}
        <div className="space-y-4">
          <Label className="text-lg font-semibold">Basic Details</Label>

          <div className="space-y-2">
            <Label htmlFor="title">Title *</Label>
            <Input
              id="title"
              type="text"
              placeholder="Enter ticket title"
              value={formData.title}
              onChange={(e) => handleInputChange("title", e.target.value)}
              disabled={isSubmitting}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description *</Label>
            <Textarea
              id="description"
              placeholder="Enter detailed description of your issue"
              rows={6}
              value={formData.description}
              onChange={(e) => handleInputChange("description", e.target.value)}
              disabled={isSubmitting}
            />
          </div>
        </div>

        <Separator />

        {/* Priority & Category */}
        <div className="space-y-4">
          <Label className="text-lg font-semibold">Priority & Category</Label>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="priority">Priority *</Label>
              <Select
                value={formData.priority}
                onValueChange={(value) => handleInputChange("priority", value)}
                disabled={isSubmitting}
              >
                <SelectTrigger id="priority">
                  <SelectValue placeholder="Select priority" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Priority Levels</SelectLabel>
                    {priorityLevels.map((priority) => (
                      <SelectItem key={priority.value} value={priority.value}>
                        {priority.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="category">Category *</Label>
              <Select
                value={formData.category}
                onValueChange={(value) => handleInputChange("category", value)}
                disabled={isSubmitting}
              >
                <SelectTrigger id="category">
                  <SelectValue placeholder="Select category" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>Categories</SelectLabel>
                    {categories.map((category) => (
                      <SelectItem key={category.value} value={category.value}>
                        {category.label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="subcategory">Subcategory (Optional)</Label>
            <Input
              id="subcategory"
              type="text"
              placeholder="Enter subcategory"
              value={formData.subcategory}
              onChange={(e) => handleInputChange("subcategory", e.target.value)}
              disabled={isSubmitting}
            />
          </div>
        </div>

        <Separator />

        {/* Submit Button */}
        <div className="flex justify-end gap-4">
          <Button
            type="button"
            variant="outline"
            onClick={() =>
              setFormData({
                title: "",
                description: "",
                priority: "Medium",
                category: "",
                subcategory: "",
                tags: [],
              })
            }
            disabled={isSubmitting}
          >
            Reset
          </Button>

          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Creating Ticket...
              </>
            ) : (
              "Raise Ticket"
            )}
          </Button>
        </div>
      </form>
    </div>
  );
};

export default RaiseTicket;
