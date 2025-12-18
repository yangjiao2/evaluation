#!/bin/bash

if ! which streamlit &> /dev/null; then
    read -p "Streamlit is not installed. Do you want to install it? (y/n): " choice
    if [[ $choice == [Yy]* ]]; then
        pip install streamlit
    else
        echo "Streamlit is required to run this script."
        exit 1
    fi
fi

streamlit run check_dataset.py
