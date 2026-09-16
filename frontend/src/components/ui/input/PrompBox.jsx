"use client";

import React, { useState, useRef } from "react";
import { Card } from "../card";
import { Button } from "../button";
import { Send, Paperclip, X, FileText, Image, File } from "lucide-react";

const PrompBox = ({
  onSend,
  placeholder = "Type your message...",
  disabled = false,
  maxFiles = 5,
  maxFileSize = 10 * 1024 * 1024, // 10MB
  acceptedTypes = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
    "text/plain",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ],
  message,
  setMessage,
}) => {
  const [attachments, setAttachments] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  // Get file icon based on file type
  const getFileIcon = (fileType) => {
    if (fileType.startsWith("image/")) {
      return <Image className="h-4 w-4 text-blue-500" />;
    } else if (fileType === "application/pdf") {
      return <FileText className="h-6 w-6 text-chart-1" />;
    } else {
      return <File className="h-4 w-4 text-gray-500" />;
    }
  };

  // Validate file
  const validateFile = (file) => {
    if (!acceptedTypes.includes(file.type)) {
      return `File type ${file.type} is not supported`;
    }
    if (file.size > maxFileSize) {
      return `File size exceeds ${formatFileSize(maxFileSize)} limit`;
    }
    return null;
  };

  // Handle file selection
  const handleFileSelect = (files) => {
    const fileArray = Array.from(files);
    const newAttachments = [];
    let hasError = false;

    // Check total file count
    if (attachments.length + fileArray.length > maxFiles) {
      setError(`Maximum ${maxFiles} files allowed`);
      setTimeout(() => setError(""), 3000);
      return;
    }

    fileArray.forEach((file) => {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        hasError = true;
        return;
      }

      // Check for duplicates
      const isDuplicate = attachments.some(
        (attachment) =>
          attachment.name === file.name && attachment.size === file.size
      );

      if (!isDuplicate) {
        newAttachments.push({
          id: Date.now() + Math.random(),
          file: file,
          name: file.name,
          size: file.size,
          type: file.type,
          preview: file.type.startsWith("image/")
            ? URL.createObjectURL(file)
            : null,
        });
      }
    });

    if (hasError) {
      setTimeout(() => setError(""), 3000);
      return;
    }

    setAttachments((prev) => [...prev, ...newAttachments]);
    setError("");
  };

  // Handle file input change
  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files);
    }
    // Reset input value to allow selecting the same file again
    e.target.value = "";
  };

  // Remove attachment
  const removeAttachment = (id) => {
    setAttachments((prev) => {
      const updated = prev.filter((attachment) => attachment.id !== id);
      // Cleanup preview URLs
      const removedAttachment = prev.find((attachment) => attachment.id === id);
      if (removedAttachment?.preview) {
        URL.revokeObjectURL(removedAttachment.preview);
      }
      return updated;
    });
  };

  // Handle drag and drop
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelect(files);
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="space-y-2">
      {/* Attachments Preview */}
      {attachments.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-sans text-muted-foreground">
            Attachments ({attachments.length}/{maxFiles})
          </div>
          <div className="flex gap-2 overflow-x-scroll no-scrollbar">
            {attachments.map((attachment) => (
              <Card
                key={attachment.id}
                className="p-2 flex flex-row max-w-[300px] items-center"
              >
                <div className="flex items-center space-x-3 flex-1 min-w-0">
                  {attachment.preview ? (
                    <img
                      src={attachment.preview}
                      alt={attachment.name}
                      className="w-8 h-8 object-cover rounded border"
                    />
                  ) : (
                    getFileIcon(attachment.type)
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {attachment.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(attachment.size)}
                    </p>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => removeAttachment(attachment.id)}
                  className="hover:bg-muted cursor-pointer"
                >
                  <X className="h-4 w-4 text-destructive" />
                </Button>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Main Input Card */}
      <Card
        className={`p-0 `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col space-x-2 p-3">
          <div className="flex-1">
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={acceptedTypes.join(",")}
              onChange={handleFileInputChange}
              className="hidden"
            />

            {/* Text input area */}
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={isDragOver ? "Drop files here..." : placeholder}
              disabled={disabled}
              rows={1}
              className="w-full resize-none border-0 bg-transparent px-0 py-1 text-base placeholder:text-muted-foreground focus:outline-none focus:ring-0 min-h-[2.5rem] max-h-32"
              style={{
                scrollbarWidth: "thin",
                scrollbarColor: "#cbd5e1 transparent",
              }}
              onInput={(e) => {
                e.target.style.height = "auto";
                e.target.style.height =
                  Math.min(e.target.scrollHeight, 128) + "px";
              }}
            />
          </div>

          <div className="flex justify-between">
            {/* Attachment button */}
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || attachments.length >= maxFiles}
              className="p-2 bg-muted"
              title="Attach files"
            >
              <Paperclip className="h-4 w-4" />
            </Button>

            {/* Send button */}
            <Button
              type="button"
              size="sm"
              onClick={() => {
                onSend();
                setMessage("");
              }}
              disabled={disabled}
              className="p-2 flex justify-center items-center cursor-pointer"
              title="Send message"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default PrompBox;
