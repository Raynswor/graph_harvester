# SPDX-FileCopyrightText: 2024 Sebastian Kempf <sebastian.kempf@uni-wuerzburg.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later


# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

ENV PYTHONPATH=/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codeCOPY app.py /app/
COPY * /app/.
COPY kieta_modules /app/kieta_modules
COPY kieta_data_objs /app/kieta_data_objs

# Expose the specified port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
