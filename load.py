import json

def load_my_json_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded data from '{file_path}'.")
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check if the file contains valid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return None
json_file_name = 'user-wallet-transactions.json' 
my_loaded_data = load_my_json_data('user-wallet-transactions.json')
if my_loaded_data is not None:
    print("\n--- Here is your loaded JSON data (first 500 characters or full if smaller) ---")

    if isinstance(my_loaded_data, (dict, list)):
        print(json.dumps(my_loaded_data, indent=2)[:500] + ("..." if len(json.dumps(my_loaded_data, indent=2)) > 500 else ""))
    else:

        print(str(my_loaded_data)[:500] + ("..." if len(str(my_loaded_data)) > 500 else ""))
        if isinstance(my_loaded_data, list):
         for item in my_loaded_data:
          print(item)
        else:
         print("\nCould not load JSON data. Please check the file path and content.")
         