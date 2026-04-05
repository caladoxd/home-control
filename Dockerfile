FROM node:20-alpine

WORKDIR /app

COPY . .

RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

RUN npm run build --if-present

ENV NODE_ENV=production

EXPOSE 3000

CMD ["npm", "start"]
