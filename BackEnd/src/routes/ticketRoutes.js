import express from "express";
import ticketController from "../controllers/ticketController.js";
import authMiddleware from "../middlewares/authMiddleware.js";

const router = express.Router();

router
  .route("/")
  .get(authMiddleware.protect, ticketController.getTickets)
  .post(authMiddleware.protect, ticketController.createTicket);

router
  .route("/my-tickets")
  .get(authMiddleware.protect, ticketController.getMyTickets);

router
  .route("/:id")
  .get(authMiddleware.protect, ticketController.getTicketById)
  .put(authMiddleware.protect, ticketController.updateTicket)
  .delete(authMiddleware.protect, ticketController.deleteTicket);

export default router;
