import express from "express";
import userController from "../controllers/userController.js";
import authMiddleware from "../middlewares/authMiddleware.js";

const router = express.Router();

// Note: The authMiddleware has a bug with an undefined 'catchAsync'.
// You will need to fix that middleware for these routes to work.

router
  .route("/")
  .post(userController.createUser) // Public route for registration
  .get(authMiddleware.protect, userController.getAllUsers); // Protected

router.route("/me").get(authMiddleware.protect, userController.getMe); // Protected route

router
  .route("/:id")
  .get(authMiddleware.protect, userController.getUserById)
  .put(authMiddleware.protect, userController.updateUser)
  .delete(authMiddleware.protect, userController.deleteUser);

export default router;
