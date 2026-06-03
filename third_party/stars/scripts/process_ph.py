import argparse
import json
import tqdm
import os
import sys

# Default singer information configuration
DEFAULT_SINGER2INFO = {
    "ZH-Alto-1": {'language': 'Chinese', 'gender': 'female'},
    "ZH-Tenor-1": {'language': 'Chinese', 'gender': 'male'},
    "EN-Alto-1": {'language': 'English', 'gender': 'female'},
    "EN-Alto-2": {'language': 'English', 'gender': 'female'},
    "EN-Tenor-1": {'language': 'English', 'gender': 'male'}
}

def process_metadata(input_file, output_file, singer_info=None):
    """
    Process metadata file and generate new processed metadata file
    
    Args:
        input_file: Path to input metadata.json file
        output_file: Path to output metadata_processed.json file
        singer_info: Dictionary of singer information, uses default if None
    """
    # Load singer information
    singer2info = singer_info or DEFAULT_SINGER2INFO
    
    # Load metadata
    try:
        items = json.load(open(input_file, 'r'))
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        sys.exit(1)
    
    new_items = []
    error_count = 0
    missing_files = []
    
    for item in tqdm.tqdm(items, desc="Processing metadata"):
        wav_fn = item['wav_fn']
        item_name = item['item_name']
        
        # Check for required keys
        required_keys = ['ph', 'ph2words', 'ph_durs', 'ep_types']
        if not all(key in item for key in required_keys):
            print(f"Warning: Missing required keys in item {item_name}")
            error_count += 1
            continue
            
        ph = item['ph']
        ph2words = item['ph2words']
        ph_durs = item['ph_durs']
        ep_types = item['ep_types']
        
        # Validate array lengths
        if not (len(ph) == len(ph2words) == len(ph_durs) == len(ep_types)):
            print(f"Error in {item_name}: Phone arrays have inconsistent lengths")
            error_count += 1
            continue
        
        # Initialize processed data lists
        gt_ph = []
        gt_ph2words = []
        gt_ph_durs = []
        note_num = []
        
        # Initialize technique feature lists
        tech_features = {
            'mix_tech': [],
            'falsetto_tech': [],
            'breathy_tech': [],
            'bubble_tech': [],
            'strong_tech': [],
            'weak_tech': [],
            'vibrato_tech': [],
            'pharyngeal_tech': [],
            'glissando_tech': []
        }
        
        # Process each phone
        i = 0
        while i < len(ep_types):
            # Merge condition: consecutive identical phones in same word
            if (i != 0 and 
                ep_types[i] == 3 and 
                ep_types[i-1] == 3 and 
                ph[i] == ph[i-1] and 
                ph2words[i] == ph2words[i-1]):
                
                # Merge with previous phone
                gt_ph_durs[-1] += ph_durs[i]
                note_num[-1] += 1
            else:
                # Add new entry
                gt_ph.append(ph[i])
                gt_ph2words.append(ph2words[i])
                gt_ph_durs.append(ph_durs[i])
                note_num.append(1)
                
                # Add technique features if they exist
                for feature in tech_features:
                    if feature in item:
                        tech_features[feature].append(item[feature][i])
            
            i += 1
        
        # Update processed data
        item['gt_ph'] = gt_ph
        item['gt_ph2words'] = gt_ph2words
        item['gt_ph_durs'] = gt_ph_durs
        item['note_num'] = note_num
        
        # Update technique features
        for feature in tech_features:
            if feature in item:
                item[f'gt_{feature}'] = tech_features[feature]
        
        # Add singer information
        singer = item['singer']
        if singer not in singer2info:
            print(f"Warning: Unknown singer '{singer}' in item {item_name}")
            item['language'] = 'Unknown'
            item['gender'] = 'Unknown'
        else:
            item['language'] = singer2info[singer]['language']
            item['gender'] = singer2info[singer]['gender']
        
        # Check if audio file exists
        if not os.path.exists(item['wav_fn']):
            missing_files.append(item['wav_fn'])
        
        new_items.append(item)
    
    # Print processing summary
    print(f"\nProcessing completed:")
    print(f"- Processed items: {len(new_items)}")
    print(f"- Errors encountered: {error_count}")
    
    if missing_files:
        print(f"- Missing audio files: {len(missing_files)}")
        for fn in missing_files[:3]:  # Show max 3 missing files
            print(f"  {fn}")
        if len(missing_files) > 3:
            print(f"  ... and {len(missing_files)-3} more")
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_items, f, ensure_ascii=False, indent=4)
    
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process singing voice metadata')
    parser.add_argument('--input', '-i', required=True, default="data/processed/chinese/metadata.json",
                        help='Input metadata.json file path')
    parser.add_argument('--output', '-o', default="data/processed/chinese/metadata_processed.json",
                        help='Output metadata_processed.json file path')
    parser.add_argument('--singer_config', '-c', 
                        help='Optional JSON file with singer information')
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        input_dir = os.path.dirname(args.input)
        args.output = os.path.join(input_dir, 'metadata_processed.json')
    
    # Load custom singer config if provided
    singer_info = None
    if args.singer_config:
        try:
            with open(args.singer_config, 'r') as f:
                singer_info = json.load(f)
            print(f"Loaded singer config from: {args.singer_config}")
        except Exception as e:
            print(f"Error loading singer config: {e}")
            sys.exit(1)
    
    # Process metadata
    process_metadata(args.input, args.output, singer_info)