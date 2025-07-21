import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df_raw):
    """
    Apply full preprocessing pipeline similar to app.py/notebook approach
    
    Parameters:
    -----------
    df_raw : pandas.DataFrame
        Raw dataframe from CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Processed dataframe with all features properly extracted and cleaned
    """
    print(f"Original dataset: {len(df_raw)} rows")
    
    # Create a copy to avoid modifying the original
    df = df_raw.copy()

    df = df.fillna(0)
    
    if "Num_bedroom" in df.columns:
        df = df.astype({"Num_bedroom": int})
    if "Num_WC" in df.columns:
        df = df.astype({"Num_WC": int})

    # STEP 1: Handle Price
    print("Processing Price...")
    df = process_price(df)
    
    # STEP 2: Process Num_bedroom
    print("Processing Num_bedroom...")
    df = process_bedrooms(df)
    # Remove rows with no bedroom info
    df = df[df["Num_bedroom"] != 0].reset_index(drop=True)
    print(f"After bedroom processing: {len(df)} rows")
    
    # STEP 3: Process Num_WC
    print("Processing Num_WC...")
    df = process_bathrooms(df)
    # Remove rows with no bathroom info
    df = df[df["Num_WC"] != 0].reset_index(drop=True)
    print(f"After bathroom processing: {len(df)} rows")
    
    # STEP 4: Extract Num_floor
    print("Processing Num_floor...")
    df = extract_floors(df)
    # Remove rows with no floor info
    df = df[df["Num_floor"] != 0].reset_index(drop=True)
    print(f"After floor processing: {len(df)} rows")
    
    # STEP 5: Handle Acreage
    print("Processing Acreage...")
    df = process_acreage(df)
    # Remove rows with missing acreage
    df = df[df["Acreage"].notna()].reset_index(drop=True)
    print(f"After acreage processing: {len(df)} rows")
    
    # Remove unnecessary columns
    if 'Name' in df.columns: df = df.drop('Name', axis=1)
    if 'Address' in df.columns: df = df.drop('Address', axis=1)
    if 'Description' in df.columns: df = df.drop('Description', axis=1)
    if 'Link' in df.columns: df = df.drop('Link', axis=1)
    
    print(f"Final processed dataset: {len(df)} rows")
    return df

def process_price(df):
    """Process and clean the Price column"""
    pd.options.display.max_colwidth = 1000
    fixed_price = []

    for x in df["Price"]:
        # Handle non-string values
        if not isinstance(x, str):
            fixed_price.append(float(x) if isinstance(x, (int, float)) else "")
            continue
        
        # Normalize price string for easier processing
        x_lower = x.lower().strip()
        
        if re.search(r"(\d+)\s+tỷ\s+(\d+)\s+triệu", x_lower) != None:
            match = re.search(r"(\d+)\s+tỷ\s+(\d+)\s+triệu", x_lower)
            billions = float(match.group(1))
            millions = float(match.group(2)) / 1000  # convert millions to billions
            fixed_price.append(billions + millions)
            
        # Handle format "X tỷ Y" (billions and millions)
        elif re.search(r"(\d+) tỷ (\d+)", x_lower) != None:
            match = re.search(r"(\d+) tỷ (\d+)", x_lower)
            billions = float(match.group(1))
            millions = float(match.group(2)) / 1000  # chuyển đơn vị triệu thành tỷ
            fixed_price.append(billions + millions)
        
        # Handle format with "triệu" (millions)
        elif re.search(" triệu", x_lower) != None:
            fixed_price.append(float(x.split(" triệu")[0]) / 1000)
        
        # Handle special cases with "X" like "8X TỶ", "8.X TỶ", "4.x TỶ"
        elif re.search(r"(\d+)[\.,]?[xX]", x_lower) != None and any(term in x_lower for term in ["tỷ", "tỉ", "ty", "ti"]):
            match = re.search(r"(\d+)([\.,][xX])?", x_lower)
            if match:
                base_num = float(match.group(1))
                fixed_price.append(base_num + 0.5)  # Using 0.5 as approximation for X
        
        # Handle format with "xíu tỷ" like "8.xíu tỷ"
        elif "xíu" in x_lower and any(term in x_lower for term in ["tỷ", "tỉ", "ty", "ti"]):
            match = re.search(r"(\d+[\.,]?\d*)", x_lower)
            if match:
                base_num = float(re.sub(",", ".", match.group(1)))
                fixed_price.append(base_num + 0.3)  # "xíu" means "a little bit", using 0.3
        
        # Handle cases like "8.3 Tỉ", "2.36 tỉ", "1,5Ty" without spaces
        elif any(term in x_lower for term in ["tỷ", "tỉ", "ty", "ti"]):
            # Extract numbers before "tỷ"/"tỉ"
            match = re.search(r"(\d+[\.,]?\d*)", x_lower)
            if match:
                value = re.sub(",", ".", match.group(1))
                fixed_price.append(float(value))
            else:
                fixed_price.append("")
        else:
            fixed_price.append("")

    df["Price"] = fixed_price

    # Check descriptions for additional price info
    checked_price = []

    for i in range(0, df.shape[0]):
        new = str(df[i:i+1]["Description"]).lower()
        
        # Handle "xíu tỷ" pattern in Description
        if "xíu" in new and any(term in new for term in ["tỷ", "tỉ", "ty", "ti"]):
            match = re.search(r"(\d+[\.,]?\d*)\s*xíu", new)
            if match:
                base_num = float(re.sub(",", ".", match.group(1)))
                checked_price.append(base_num + 0.3)
                continue
                
        # Handle X pattern (8X TỶ, 2.X TỶ)
        x_pattern = re.search(r"(\d+[\.,]?)[xX]", new)
        if x_pattern and any(term in new for term in ["tỷ", "tỉ", "ty", "ti"]):
            base_num = float(re.sub(",", ".", x_pattern.group(1)))
            checked_price.append(base_num + 0.5)
            continue
        
        # Original patterns for price detection
        if re.search(r"(\d*\.\d+|\d*\,\d+|\d+) tỷ|(\d*\.\d+|\d*\,\d+|\d+) TỶ|(\d*\.\d+|\d*\,\d+|\d+) Tỷ|(\d*\.\d+|\d*\,\d+|\d+) ty", new) == None:
            if re.search(r"(\d*\.\d+|\d*\,\d+|\d+)tỷ|(\d*\.\d+|\d+)Tỷ|(\d*\.\d+|\d*\,\d+|\d+)TỶ|(\d*\.\d+|\d*\,\d+|\d+)ty", new) == None: 
                checked_price.append("")
            else: 
                checked_price.append(float(re.sub(",", ".",re.findall(r"[0-9]+.+[0-9]|\d+",re.search(r"(\d*\.\d+|\d*\,\d+|\d+)tỷ|(\d*\.\d+|\d*\,\d+|\d+)Tỷ|(\d*\.\d+|\d*\,\d+|\d+)TỶ|(\d*\.\d+|\d*\,\d+|\d+)ty", new).group())[0])))
        else: 
            checked_price.append(float(re.sub(",", ".",re.findall(r"[0-9]+.+[0-9]|\d+",re.search(r"(\d*\.\d+|\d*\,\d+|\d+) tỷ|(\d*\.\d+|\d*\,\d+|\d+) TỶ|(\d*\.\d+|\d*\,\d+|\d+) Tỷ|(\d*\.\d+|\d*\,\d+|\d+) ty", new).group())[0])))

    # Also check price patterns in Name field
    for i in range(0, df.shape[0]):
        new = str(df.iloc[i:i+1]["Name"]).lower()
        price_found = False  # Flag to track if we've found a price
        
        ty_pattern = re.search(r"(\d+)\s*(tỷ|tỉ|ty|ti)\s+(\d+)", new, re.IGNORECASE)
        if ty_pattern:
            billions = float(ty_pattern.group(1))
            millions_part = float(ty_pattern.group(3))
            if millions_part < 1000:
                millions = millions_part / 1000  # convert to billions (assuming the number after is in millions)
                checked_price[i] = billions + millions
                price_found = True
                
        # Handle "xíu tỷ" in Name
        if "xíu" in new and any(term in new for term in ["tỷ", "tỉ", "ty", "ti"]):
            match = re.search(r"(\d+[\.,]?\d*)\s*xíu", new)
            if match:
                base_num = float(re.sub(",", ".", match.group(1)))
                checked_price[i] = base_num + 0.3
                price_found = True
        
        # Xử lý trường hợp "5.x tỷ", "5,x tỷ"
        if not price_found:
            dot_x_pattern = re.search(r"(\d+)[\.,][xX] tỷ|(\d+)[\.,][xX]tỷ|(\d+)[\.,][xX] Tỷ|(\d+)[\.,][xX]Tỷ", new)
            if dot_x_pattern:
                # Lấy số trước dấu chấm/phẩy và chuyển thành giá
                base_num = float(re.search(r"(\d+)", dot_x_pattern.group()).group(1))
                checked_price[i] = base_num + 0.5  # 5.x tỷ = ~5.5 tỷ (giá trị trung bình)
                price_found = True
        
        # Xử lý trường hợp "5x tỷ", "5X tỷ"
        if not price_found:
            x_pattern = re.search(r"(\d+)[xX] tỷ|(\d+)[xX]tỷ|(\d+)[xX] Tỷ|(\d+)[xX]Tỷ", new)
            if x_pattern:
                # Lấy số trước x và chuyển thành giá
                base_num = float(re.search(r"(\d+)", x_pattern.group()).group(1))
                checked_price[i] = base_num + 0.5  # Giả sử x đại diện cho .5 (ví dụ 5x = 5.5)
                price_found = True
        
        if not price_found:
            no_space_pattern = re.search(r"(\d+)(tỷ|tỉ|ty|ti|TỶ|TỈ|TY|TI)", new)
            if no_space_pattern:
                billions = float(no_space_pattern.group(1))
                if billions <= 500:  
                    checked_price[i] = billions
                    price_found = True
        
        # Handle cases like "8.5TỈ", "10,5TỶ" with decimal but no spaces
        if not price_found:
            decimal_no_space = re.search(r"(\d+[\.,]\d+)(tỷ|tỉ|ty|ti|TỶ|TỈ|TY|TI)", new)
            if decimal_no_space:
                checked_price[i] = float(re.sub(",", ".", decimal_no_space.group(1)))
                price_found = True
                
        # Xử lý các trường hợp bình thường nếu chưa tìm được giá
        if not price_found:
            # Add patterns for "1,5Ty", "8.3 Tỉ", etc. without spaces or with different formats
            if re.search(r"(\d*[\.,]\d+|\d+)[\s]*(tỷ|tỉ|ty|ti)", new, re.IGNORECASE):
                match = re.search(r"(\d*[\.,]\d+|\d+)[\s]*(tỷ|tỉ|ty|ti)", new, re.IGNORECASE)
                checked_price[i] = float(re.sub(",", ".", match.group(1)))
                price_found = True
                
            # Original patterns
            elif not price_found:
                if re.search(r"(\d*\.\d+|\d*\,\d+|\d+) tỷ|(\d*\.\d+|\d*\,\d+|\d+) TỶ|(\d*\.\d+|\d*\,\d+|\d+) Tỷ|(\d*\.\d+|\d*\,\d+|\d+) ty", new) == None:
                    if re.search(r"(\d*\.\d+|\d*\,\d+|\d+)tỷ|(\d*\.\d+|\d+)Tỷ|(\d*\.\d+|\d*\,\d+|\d+)TỶ|(\d*\.\d+|\d*\,\d+|\d+)ty", new) == None: 
                        continue
                    else: 
                        checked_price[i] = float(re.sub(",", ".",re.findall(r"[0-9]+.+[0-9]|\d+",re.search(r"(\d*\.\d+|\d*\,\d+|\d+)tỷ|(\d*\.\d+|\d*\,\d+|\d+)Tỷ|(\d*\.\d+|\d*\,\d+|\d+)TỶ|(\d*\.\d+|\d*\,\d+|\d+)ty", new).group())[0]))
                else: 
                    checked_price[i] = float(re.sub(",", ".",re.findall(r"[0-9]+.+[0-9]|\d+",re.search(r"(\d*\.\d+|\d*\,\d+|\d+) tỷ|(\d*\.\d+|\d*\,\d+|\d+) TỶ|(\d*\.\d+|\d*\,\d+|\d+) Tỷ|(\d*\.\d+|\d*\,\d+|\d+) ty", new).group())[0]))

    # Update Price based on checked prices
    for i in range(0, df.shape[0]):
        price_at_i = df.at[i, "Price"]
        if checked_price[i] != "" and (price_at_i == "" or 
                                     (price_at_i != "" and 
                                     ((price_at_i == 0) or  
                                     (price_at_i != 0 and (checked_price[i] / price_at_i > 10 or checked_price[i] / price_at_i < 0.1))))):
            df.at[i, "Price"] = checked_price[i]

    # Convert to numeric and filter out extreme values
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Calculate mean for missing values - only to replace empty values
    mean_price = df[df["Price"] > 0]["Price"].mean()
    
    # Replace zeros with mean price
    zero_count = (df["Price"] == 0).sum()
    df.loc[df["Price"] == 0, "Price"] = mean_price
    
    # For any NaN values (previously strings), also replace with the mean
    df.loc[df["Price"].isna(), "Price"] = mean_price
    
    # Filter out rows with missing or extreme prices
    df = df[df["Price"] != ""]
    original_count = len(df)
    
    df = df[df["Price"] <= 500].reset_index(drop=True)
    
    print(f"After price processing: {len(df)} rows (removed {original_count - len(df)} extreme prices)")
    
    return df

def process_bedrooms(df):
    """Process and extract Num_bedroom without filling missing values"""
    if 'Num_bedroom' not in df.columns:
        df['Num_bedroom'] = 0
    else:
        df['Num_bedroom'] = df['Num_bedroom'].fillna(0).astype(int)
    
    for i in range(0, df.shape[0]):
        if df.at[i,"Num_bedroom"] != 0: 
            continue
        new = str(df.iloc[i:i+1]["Description"])
        if re.search(r"(\d+) phòng ngủ|(\d+) PN|(\d+) pn|(\d+) Pn|(\d+) ngủ|(\d+) p ngủ|(\d+) Phòng ngủ|(\d+) Phòng Ngủ|(\d+) P Ngủ", new) == None:
            if re.search(r"(\d+)PN|(\d+)pn|(\d+)Pn|(\d+)ngủ|(\d+)p ngủ|(\d+)Phòng ngủ|(\d+)Phòng Ngủ|(\d+)phòng ngủ|(\d+)P Ngủ", new) == None: 
                continue
            else: 
                df.at[i,"Num_bedroom"] = int(re.findall(r"[0-9]+",re.search(r"(\d+)PN|(\d+)pn|(\d+)Pn|(\d+)ngủ|(\d+)p ngủ|(\d+)Phòng ngủ|(\d+)Phòng Ngủ|(\d+)phòng ngủ|(\d+)P Ngủ", new).group())[0])
        else: 
            df.at[i,"Num_bedroom"] = int(re.search(r"(\d+) phòng ngủ|(\d+) PN|(\d+) pn|(\d+) Pn|(\d+) ngủ|(\d+) p ngủ|(\d+) Phòng ngủ|(\d+) Phòng Ngủ|(\d+) P Ngủ", new).group().split(" ")[0])

    # Extract from Name
    for i in range(0, df.shape[0]):
        if df.at[i,"Num_bedroom"] != 0: 
            continue
        new = str(df.iloc[i:i+1]["Name"])
        if re.search(r"(\d+) phòng ngủ|(\d+) PN|(\d+) pn|(\d+) Pn|(\d+) ngủ|(\d+) p ngủ|(\d+) Phòng ngủ|(\d+) Phòng Ngủ|(\d+) P Ngủ", new) == None:
            if re.search(r"(\d+)PN|(\d+)pn|(\d+)Pn|(\d+)ngủ|(\d+)p ngủ|(\d+)Phòng ngủ|(\d+)Phòng Ngủ|(\d+)phòng ngủ|(\d+)P Ngủ", new) == None: 
                continue
            else: 
                df.at[i,"Num_bedroom"] = int(re.findall(r"[0-9]+",re.search(r"(\d+)PN|(\d+)pn|(\d+)Pn|(\d+)ngủ|(\d+)p ngủ|(\d+)Phòng ngủ|(\d+)Phòng Ngủ|(\d+)phòng ngủ|(\d+)P Ngủ", new).group())[0])
        else: 
            df.at[i,"Num_bedroom"] = int(re.search(r"(\d+) phòng ngủ|(\d+) PN|(\d+) pn|(\d+) Pn|(\d+) ngủ|(\d+) p ngủ|(\d+) Phòng ngủ|(\d+) Phòng Ngủ|(\d+) P Ngủ", new).group().split(" ")[0])

    df['Num_bedroom'] = df['Num_bedroom'].apply(lambda x: 1 if x == 0 else x)

    return df

def process_bathrooms(df):
    """Process and extract Num_WC without filling missing values"""
    if 'Num_WC' not in df.columns:
        df['Num_WC'] = 0
    else:
        df['Num_WC'] = df['Num_WC'].fillna(0).astype(int)
    
    # Extract from Description and Name using existing code
    # Extract from Description
    for i in range(0, df.shape[0]):
        if df.at[i,"Num_WC"] != 0: 
            continue
        new = str(df.iloc[i:i+1]["Description"])
        if re.search(r"(\d+) toilet|(\d+) Toilet|(\d+) wc|(\d+) Wc|(\d+) WC|(\d+) vệ sinh|(\d+) phòng vệ sinh|(\d+) nhà vệ sinh|(\d+) Nhà vệ sinh|(\d+) Vệ sinh|(\d+) VS", new) == None:
            if re.search(r"(\d+)toilet|(\d+)Toilet|(\d+)wc|(\d+)Wc|(\d+)WC|(\d+)vệ sinh|(\d+)phòng vệ sinh|(\d+)nhà vệ sinh|(\d+)Nhà vệ sinh|(\d+)Vệ sinh|(\d+)VS", new) == None:
                if re.search(r"toilet|wc|Wc|WC", new) == None: 
                    continue
                else: 
                    df.at[i,"Num_WC"] = len(re.findall(r"toilet|wc|Wc|WC", new))
            else: 
                df.at[i,"Num_WC"] = int(re.findall(r"[0-9]+",re.search(r"(\d+)toilet|(\d+)Toilet|(\d+)wc|(\d+)Wc|(\d+)WC|(\d+)vệ sinh|(\d+)phòng vệ sinh|(\d+)nhà vệ sinh|(\d+)Nhà vệ sinh|(\d+)Vệ sinh|(\d+)VS", new).group())[0])
        else: 
            df.at[i,"Num_WC"] = int(re.search(r"(\d+) toilet|(\d+) Toilet|(\d+) wc|(\d+) Wc|(\d+) WC|(\d+) vệ sinh|(\d+) phòng vệ sinh|(\d+) nhà vệ sinh|(\d+) Nhà vệ sinh|(\d+) Vệ sinh|(\d+) VS", new).group().split(" ")[0])

    # Extract from Name
    for i in range(0, df.shape[0]):
        if df.at[i,"Num_WC"] != 0: 
            continue
        new = str(df.iloc[i:i+1]["Name"])
        if re.search(r"(\d+) toilet|(\d+) Toilet|(\d+) wc|(\d+) Wc|(\d+) WC|(\d+) vệ sinh|(\d+) phòng vệ sinh|(\d+) nhà vệ sinh|(\d+) Nhà vệ sinh|(\d+) Vệ sinh|(\d+) VS", new) == None:
            if re.search(r"(\d+)toilet|(\d+)Toilet|(\d+)wc|(\d+)Wc|(\d+)WC|(\d+)vệ sinh|(\d+)phòng vệ sinh|(\d+)nhà vệ sinh|(\d+)Nhà vệ sinh|(\d+)Vệ sinh|(\d+)VS", new) == None: 
                continue
            else: 
                df.at[i,"Num_WC"] = int(re.findall(r"[0-9]+",re.search(r"(\d+)toilet|(\d+)Toilet|(\d+)wc|(\d+)Wc|(\d+)WC|(\d+)vệ sinh|(\d+)phòng vệ sinh|(\d+)nhà vệ sinh|(\d+)Nhà vệ sinh|(\d+)Vệ sinh|(\d+)VS", new).group())[0])
        else: 
            df.at[i,"Num_WC"] = int(re.search(r"(\d+) toilet|(\d+) Toilet|(\d+) wc|(\d+) Wc|(\d+) WC|(\d+) vệ sinh|(\d+) phòng vệ sinh|(\d+) nhà vệ sinh|(\d+) Nhà vệ sinh|(\d+) Vệ sinh|(\d+) VS", new).group().split(" ")[0])

    df['Num_WC'] = df['Num_WC'].apply(lambda x: 1 if x == 0 else x)

    return df

def extract_floors(df):
    """Extract Num_floor without filling missing values"""
    new_col = []
    
    for i in range(0, df.shape[0]):
        new = str(df.iloc[i:i+1]["Description"])
        if re.search(r"(\d+) tầng|(\d+) Tầng|(\d+) TẦNG", new) == None:
            if re.search(r"(\d+)tầng|(\d+)Tầng|(\d+)TẦNG", new) == None:
                if re.search(r"(\d+) lầu|(\d+) Lầu|(\d+) LẦU|(\d+) L |(\d+) sàn lầu", new) == None:
                    if re.search(r"(\d+)lầu|(\d+)Lầu|(\d+)LẦU|(\d+)L |(\d+)sàn lầu", new) == None: 
                        new_col.append(0)
                    else: 
                        new_col.append(int(re.findall(r"[0-9]+",re.search(r"(\d+)lầu|(\d+)Lầu|(\d+)LẦU|(\d+)L |(\d+)sàn lầu", new).group())[0]) + 1)
                else: 
                    new_col.append(int(re.search(r"(\d+) lầu|(\d+) Lầu|(\d+) LẦU|(\d+) L |(\d+) sàn lầu", new).group().split(" ")[0]) + 1)
            else: 
                new_col.append(int(re.findall(r"[0-9]+",re.search(r"(\d+)tầng|(\d+)Tầng|(\d+)TẦNG", new).group())[0]))
        else: 
            new_col.append(int(re.search(r"(\d+) tầng|(\d+) Tầng|(\d+) TẦNG", new).group().split(" ")[0]))

    # Add new column to dataframe
    if 'Num_floor' not in df.columns:
        df.insert(4, "Num_floor", new_col, True)

    # Extract from Name
    for i in range(0, df.shape[0]):
        if df.at[i,"Num_floor"] != 0: 
            continue
        new = str(df.iloc[i:i+1]["Name"])
        if re.search(r"(\d+) tầng|(\d+) Tầng|(\d+) TẦNG", new) == None:
            if re.search(r"(\d+)tầng|(\d+)Tầng|(\d+)TẦNG", new) == None:
                if re.search(r"(\d+) lầu|(\d+) Lầu|(\d+) LẦU|(\d+) L |(\d+) sàn lầu", new) == None:
                    if re.search(r"(\d+)lầu|(\d+)Lầu|(\d+)LẦU|(\d+)L |(\d+)sàn lầu", new) == None: 
                        continue
                    else: 
                        df.at[i,"Num_floor"] = int(re.findall(r"[0-9]+",re.search(r"(\d+)lầu|(\d+)Lầu|(\d+)LẦU|(\d+)L |(\d+)sàn lầu", new).group())[0]) + 1
                else: 
                    df.at[i,"Num_floor"] = int(re.search(r"(\d+) lầu|(\d+) Lầu|(\d+) LẦU|(\d+) L |(\d+) sàn lầu", new).group().split(" ")[0]) + 1
            else: 
                df.at[i,"Num_floor"] = int(re.findall(r"[0-9]+",re.search(r"(\d+)tầng|(\d+)Tầng|(\d+)TẦNG", new).group())[0])
        else:
            df.at[i,"Num_floor"] = int(re.search(r"(\d+) tầng|(\d+) Tầng|(\d+) TẦNG", new).group().split(" ")[0])

    df["Num_floor"] = df["Num_floor"].replace(0, 1)

    return df

def process_acreage(df):
    """Process Acreage column without filling missing values"""
    if 'Acreage' in df.columns:
        if df['Acreage'].dtype == 'object':
            acreage_values = []
            for value in df['Acreage']:
                if pd.isna(value) or value == '':
                    acreage_values.append(np.nan)
                    continue
                    
                if isinstance(value, str):
                    match = re.search(r'(\d+(\.\d+)?)', value)
                    if match:
                        acreage_values.append(float(match.group(1)))
                    else:
                        acreage_values.append(np.nan)
                else:
                    acreage_values.append(float(value))
            
            df['Acreage'] = acreage_values
    
    return df

def create_price_category(df):
    """Create price categories (0-3) based on price ranges"""
    conditions = [
        (df['Price'] < 1),
        (df['Price'] >= 1) & (df['Price'] < 3),
        (df['Price'] >= 3) & (df['Price'] < 7),
        (df['Price'] >= 7)
    ]
    choices = [0, 1, 2, 3]
    df['Price_Category'] = np.select(conditions, choices, default=np.nan)
    return df

def load_and_preprocess_data(file_path='data/df.csv'):
    """Load data from CSV and apply full preprocessing pipeline"""
    df_raw = pd.read_csv(file_path)
    df = preprocess_data(df_raw)
    df = create_price_category(df)
    return df

def scale_features(df, features_to_scale=None):
    """Scale numeric features using StandardScaler"""
    if features_to_scale is None:
        features_to_scale = ["Acreage", "Num_floor", "Num_bedroom"]
        
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df_scaled

def prepare_train_test_split(df, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, validation and test sets"""
    # Separate features and target
    y = df["Price"]
    X = df.drop("Price", axis=1)
    
    # First split: training vs test+validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: test vs validation
    # Calculate relative validation size
    relative_val_size = val_size / test_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
