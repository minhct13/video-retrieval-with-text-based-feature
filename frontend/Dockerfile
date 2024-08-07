FROM node:16-alpine as builder
RUN mkdir /uidemo && chown node:node /uidemo
WORKDIR /uidemo
USER node
COPY --chown=node:node package.json package-lock.json* ./
RUN npm install 
COPY --chown=node:node . .
EXPOSE 80
CMD ["npm", "run", "dev"]
# RUN npm run build 

# FROM nginx:alpine
# COPY --from=builder /uidemo/dist /usr/share/nginx/html
# RUN rm /etc/nginx/conf.d/default.conf
# COPY nginx/nginx.conf /etc/nginx/conf.d
# EXPOSE 5173
# ENTRYPOINT ["nginx", "-g", "daemon off;"]

