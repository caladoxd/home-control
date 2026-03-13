FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm install --legacy-peer-deps

COPY . .
RUN npm run build || true

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=production

COPY --from=build /app/package*.json ./
RUN npm install --legacy-peer-deps --only=production
COPY --from=build /app/dist ./dist 2>/dev/null || COPY --from=build /app/build ./build 2>/dev/null || true
COPY --from=build /app/src ./src 2>/dev/null || true

EXPOSE 3000

CMD ["node", "dist/main.js"]