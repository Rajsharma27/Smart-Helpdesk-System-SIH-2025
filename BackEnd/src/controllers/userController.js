import { token } from "morgan";
import User from "../models/User.js";
import bcrypt from "bcryptjs";
import { createAndSendTokenCookie } from "../utils/authHelper.js";

const userController = {
  // @desc    Create a new user
  // @route   POST /api/users
  // @access  Public (for now, should be Admin in production)

  // Update this function for new user model
  // @desc    Register a new user
  // @route   POST /api/users/register
  // @access  Public
  createUser: async (req, res) => {
    const {
      name,
      email,
      password,
      employeeId,
      department,
      designation,
      role,
      avatarUrl,
      contact,
    } = req.body;

    try {
      // Validate required fields
      if (!name || !email || !password) {
        return res
          .status(400)
          .json({ status: "error", msg: "Please provide name, email, and password" });
      }
      let user = await User.findOne({ email });
      if (user) {
        return res
          .status(400)
          .json({ msg: "User already exists", email });
      }

      const salt = await bcrypt.genSalt(10);
      const hashedPassword = await bcrypt.hash(password, salt);

      user = new User({
        name,
        email,
        password: hashedPassword,
        employeeId,
        department,
        designation,
        role,
        avatarUrl,
        contact,
      });

      await user.save();

      // Return user without password
      const userToReturn = user.toObject();
      delete userToReturn.password;

      const token = createAndSendTokenCookie(user, res);

      res
        .status(201)
        .json({ status: "success", token, data: { user: userToReturn } });
    } catch (error) {
      console.error(error);
      res.status(500).json({ status: "error", msg: "Server Error" });
    }
  },

  getMe: async (req, res) => {
    try {
      const user = await User.findById(req.user.id).select("-password");
      res.json({ status: "success", data: { user } });
    } catch (error) {
      console.error(error);
      res.status(500).json({ status: "error", msg: "Server Error" });
    }
  },

  // @desc    Get all users
  // @route   GET /api/users
  // @access  Private/Admin
  getAllUsers: async (req, res) => {
    try {
      const users = await User.find().select("-password");
      res.json(users);
    } catch (error) {
      console.error(error);
      res.status(500).json({ status: "error", msg: "Server Error" });
    }
  },

  // @desc    Get user by ID
  // @route   GET /api/users/:id
  // @access  Private/Admin
  getUserById: async (req, res) => {
    try {
      const user = await User.findById(req.params.id).select("-password");

      if (!user) {
        return res.status(404).json({ status: "error", msg: "User not found" });
      }

      res.status(200).json({ status: "success", data: { user } });
    } catch (error) {
      console.error(error.message);
      if (error.kind === "ObjectId") {
        return res.status(404).json({ status: "error", msg: "User not found" });
      }
      res.status(500).json({ status: "error", msg: "Server Error" });
    }
  },

  // @desc    Update a user
  // @route   PUT /api/users/:id
  // @access  Private/Admin
  updateUser: async (req, res) => {
    const { name, email, password, isAdmin } = req.body;

    try {
      let user = await User.findById(req.params.id);
      if (!user) {
        return res.status(404).json({ status: "error", msg: "User not found" });
      }

      // Build user object
      const userFields = { name, email, isAdmin };
      if (password) {
        const salt = await bcrypt.genSalt(10);
        userFields.password = await bcrypt.hash(password, salt);
      }

      user = await User.findByIdAndUpdate(
        req.params.id,
        { $set: userFields },
        { new: true }
      ).select("-password");

      res.status(200).json({ status: "success", data: { user } });
    } catch (error) {
      console.error(error.message);
      res.status(500).json({ status: "error", msg: "Server Error" });
    }
  },

  // @desc    Delete a user
  // @route   DELETE /api/users/:id
  // @access  Private/Admin
  deleteUser: async (req, res) => {
    try {
      const user = await User.findById(req.params.id);

      if (!user) {
        return res.status(404).json({ status: "error", msg: "User not found" });
      }

      await user.deleteOne(); // Mongoose 6+

      res.status(200).json({ status: "success", msg: "User removed" });
    } catch (error) {
      console.error(error.message);
      if (error.kind === "ObjectId") {
        return res.status(404).json({ status: "error", msg: "User not found" });
      }
      res.status(500).json({ status: "error", msg: "Server Error" });
    }
  },
};

export default userController;
