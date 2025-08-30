# Use official Node.js image as the base image
FROM node:18-alpine

# Set working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json from Frontend directory first to leverage Docker cache
COPY Frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the Frontend application code
COPY Frontend/ ./

# Build the React app for production
RUN npm run build

# Use a lightweight web server to serve the static files
FROM nginx:alpine

# Copy built React app from previous stage
COPY --from=0 /app/build /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
