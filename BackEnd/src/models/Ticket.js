import mongoose from "mongoose";

// Placeholder for aiAnalysisSchema. Define this according to your needs.
const aiAnalysisSchema = new mongoose.Schema({
  sentiment: String,
  suggestedCategory: String,
});

const ticketSchema = new mongoose.Schema(
  {
    title: {
      type: String,
      required: true,
    },
    description: String,
    priority: {
      type: String,
      enum: ["Low", "Medium", "High"],
      default: "Low",
    },
    category: {
      type: String,
      enum: ["Password Reset", "Hardware", "Software", "Network", "Other"],
      default: "Other",
    },
    subcategory: String,
    status: {
      type: String,
      enum: ["Open", "In Progress", "Closed"],
      default: "Open",
    },
    source: {
      type: String,
      default: "Chatbot",
    },
    tags: {
      type: [String],
      default: [],
    },
    aiAnalysis: {
      type: aiAnalysisSchema,
      default: {},
    },
    createdBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    assignedTo: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
    },
  },
  {
    timestamps: true,
  }
);

const Ticket = mongoose.model("Ticket", ticketSchema);

export default Ticket;
