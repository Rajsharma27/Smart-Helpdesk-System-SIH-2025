import User from "../models/User.js";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { createAndSendTokenCookie } from "../utils/authHelper.js";

const authController = {
  // @desc    Auth user & get token
  // @route   POST /api/auth/login
  // @access  Public
  login: async (req, res) => {
    const { email, password } = req.body;

    try {
      // Check if user exists
      const user = await User.findOne({ email });

      if (!user) {
        return res.status(401).json({ msg: "Invalid credentials" });
      }

      // Check if password matches
      const isMatch = await bcrypt.compare(password, user.password);

      if (!isMatch) {
        return res.status(401).json({ msg: "Invalid credentials" });
      }

      const token = createAndSendTokenCookie(user, res);

      res.status(200).json({ status: "success", token, data: { user } });
    } catch (error) {
      console.error(error.message);
      res.status(500).json({ msg: "Server Error" });
    }
  },
};

export default authController;
