import streamlit as st
import numpy as np
import pandas as pd
from collections import deque
import math

# Set page title and configuration
st.set_page_config(page_title="Memory Paging Simulator", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .page-fault {
        background-color: #ffdddd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .present {
        color: green;
        font-weight: bold;
    }
    .not-present {
        color: red;
        font-weight: bold;
    }
    .fifo-queue {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
    }
    .binary {
        font-family: monospace;
        background-color: #f0f0f0;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .conversion-details {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title of the app
st.title("Memory Paging Simulator")

# Helper functions for binary conversions and calculations
def dec_to_bin(num_bits, decimal_num):
    """Convert decimal to binary with specified number of bits"""
    if decimal_num < 0:
        return "0" * num_bits
    binary = bin(decimal_num)[2:]  # Remove '0b' prefix
    return binary.zfill(num_bits)  # Pad with zeros to reach num_bits

def bin_to_dec(binary_str):
    """Convert binary string to decimal"""
    return int(binary_str, 2)

class MemorySimulator:
    def __init__(self):
        self.virtual_size = 16  # Default 16 KB
        self.page_size = 2      # Default 2 KB
        self.physical_size = 8  # Default 8 KB (1/2 of virtual)
        self.memory_map = []    # Page table
        self.fifo_queue = deque()  # Queue for FIFO page replacement
        self.conversion_type = "virtual_to_phys"  # Default conversion type
        
        # Calculated values
        self.total_bits = 0
        self.offset_bits = 0
        self.index_bits = 0
        self.num_virtual_pages = 0
        self.num_physical_frames = 0

    def calculate_memory_parameters(self):
        """Calculate memory parameters based on user inputs"""
        # Convert KB to bytes
        virtual_bytes = self.virtual_size * 1024
        page_bytes = self.page_size * 1024
        physical_bytes = self.physical_size * 1024
        
        # Calculate bits
        self.total_bits = math.ceil(math.log2(virtual_bytes))
        self.offset_bits = math.ceil(math.log2(page_bytes))
        self.index_bits = self.total_bits - self.offset_bits
        
        # Calculate number of pages and frames
        self.num_virtual_pages = 2 ** self.index_bits
        self.num_physical_frames = physical_bytes // page_bytes
        
        # Initialize the memory map
        self.initialize_memory_map()

    def initialize_memory_map(self):
        """Initialize the memory map (page table)"""
        self.memory_map = []
        for i in range(self.num_virtual_pages):
            self.memory_map.append({
                'virtual_index': i,
                'virtual_binary': dec_to_bin(self.index_bits, i),
                'physical_index': None,
                'physical_binary': None,
                'present': False
            })
        
        # Clear FIFO queue
        self.fifo_queue = deque()

    def update_page_table(self, virtual_page, physical_page, present):
        """Update a specific entry in the page table"""
        self.memory_map[virtual_page]['physical_index'] = physical_page
        self.memory_map[virtual_page]['physical_binary'] = dec_to_bin(self.index_bits, physical_page) if physical_page is not None else None
        self.memory_map[virtual_page]['present'] = present
        
        # Update FIFO queue if page is present
        if present and physical_page is not None:
            # Remove if already in queue (shouldn't happen in normal operation)
            if virtual_page in self.fifo_queue:
                self.fifo_queue.remove(virtual_page)
            # Add to queue
            self.fifo_queue.append(virtual_page)

    def convert_address(self, address):
        """Convert address between virtual and physical based on the current setting"""
        try:
            address_dec = int(address)
            address_bin = dec_to_bin(self.total_bits, address_dec)
            
            # Split address into page index and offset
            page_index_bin = address_bin[:self.index_bits]
            offset_bin = address_bin[self.index_bits:]
            
            page_index = bin_to_dec(page_index_bin)
            offset = bin_to_dec(offset_bin)
            
            # Results dictionary to return
            result = {
                'original_address_dec': address_dec,
                'original_address_bin': address_bin,
                'page_index_bin': page_index_bin,
                'page_index_dec': page_index,
                'offset_bin': offset_bin,
                'offset_dec': offset,
                'page_fault': False,
                'page_replaced': None,
                'converted_address_bin': None,
                'converted_address_dec': None,
                'result_status': 'success'
            }
            
            if self.conversion_type == "virtual_to_phys":
                # Check if the page is present
                if not self.memory_map[page_index]['present']:
                    # Page fault! Handle with FIFO
                    result['page_fault'] = True
                    self.handle_page_fault(page_index, result)
                
                # Get the physical frame
                physical_frame = self.memory_map[page_index]['physical_index']
                physical_frame_bin = dec_to_bin(self.index_bits, physical_frame)
                
                # Combine with offset to get physical address
                physical_address_bin = physical_frame_bin + offset_bin
                physical_address_dec = bin_to_dec(physical_address_bin)
                
                result['converted_address_bin'] = physical_address_bin
                result['converted_address_dec'] = physical_address_dec
                
            elif self.conversion_type == "phys_to_virtual":
                # In physical to virtual conversion, we need to search for the physical frame
                physical_frame_bin = page_index_bin
                physical_frame = page_index
                
                # Find which virtual page maps to this physical frame
                virtual_page = None
                for i, entry in enumerate(self.memory_map):
                    if entry['present'] and entry['physical_index'] == physical_frame:
                        virtual_page = i
                        break
                
                if virtual_page is None:
                    result['result_status'] = 'error'
                    result['error_message'] = f"No virtual page maps to physical frame {physical_frame}"
                    return result
                
                # Get the virtual address
                virtual_page_bin = dec_to_bin(self.index_bits, virtual_page)
                virtual_address_bin = virtual_page_bin + offset_bin
                virtual_address_dec = bin_to_dec(virtual_address_bin)
                
                result['converted_address_bin'] = virtual_address_bin
                result['converted_address_dec'] = virtual_address_dec
            
            return result
        
        except Exception as e:
            return {
                'result_status': 'error',
                'error_message': f"Error converting address: {str(e)}"
            }

    def handle_page_fault(self, virtual_page, result):
        """Handle page fault using FIFO replacement algorithm"""
        # Check if there's space available in physical memory
        assigned_frames = sum(1 for entry in self.memory_map if entry['present'])
        
        if assigned_frames < self.num_physical_frames:
            # There's free space, assign a new frame
            new_frame = assigned_frames
            self.update_page_table(virtual_page, new_frame, True)
        else:
            # Need to replace using FIFO
            page_to_replace = self.fifo_queue.popleft()
            replaced_frame = self.memory_map[page_to_replace]['physical_index']
            
            # Mark the replaced page as not present
            self.update_page_table(page_to_replace, None, False)
            
            # Assign the freed frame to the new page
            self.update_page_table(virtual_page, replaced_frame, True)
            
            # Record the replacement in the result
            result['page_replaced'] = page_to_replace

# Initialize the session state if it doesn't exist
if 'simulator' not in st.session_state:
    st.session_state.simulator = MemorySimulator()
    st.session_state.page_table_initialized = False
    st.session_state.address_results = []

# Function to restart the simulation
def restart_simulation():
    st.session_state.simulator = MemorySimulator()
    st.session_state.page_table_initialized = False
    st.session_state.address_results = []

# Create a sidebar for inputs
with st.sidebar:
    st.header("Memory Configuration")
    
    # Input for virtual memory size
    virtual_size = st.number_input(
        "Virtual Memory Size (KB)", 
        min_value=4, 
        value=16, 
        step=4,
        help="Size of virtual memory in KB. Must be a power of 2."
    )
    
    # Input for page size
    page_size = st.number_input(
        "Page Size (KB)", 
        min_value=1, 
        value=2, 
        step=1,
        help="Size of each page in KB. Must be a power of 2 and smaller than virtual memory size."
    )
    
    # Physical memory is automatically set to half of virtual memory
    physical_size = virtual_size // 2
    st.write(f"Physical Memory Size: {physical_size} KB (1/2 of Virtual Memory)")
    
    # Conversion type selection
    conversion_type = st.selectbox(
        "Conversion Type",
        ["virtual_to_phys", "phys_to_virtual"],
        format_func=lambda x: "Virtual to Physical" if x == "virtual_to_phys" else "Physical to Virtual"
    )
    
    # Button to initialize the memory configuration
    if st.button("Initialize Memory"):
        # Validate inputs
        valid_input = True
        error_message = ""
        
        # Check if virtual size is a power of 2
        if virtual_size & (virtual_size - 1) != 0:
            valid_input = False
            error_message += "Virtual Memory Size must be a power of 2. "
        
        # Check if page size is a power of 2
        if page_size & (page_size - 1) != 0:
            valid_input = False
            error_message += "Page Size must be a power of 2. "
        
        # Check if page size is smaller than virtual size
        if page_size >= virtual_size:
            valid_input = False
            error_message += "Page Size must be smaller than Virtual Memory Size. "
        
        # Check if virtual size is divisible by page size
        if virtual_size % page_size != 0:
            valid_input = False
            error_message += "Virtual Memory Size must be divisible by Page Size. "
            
        if valid_input:
            # Set memory parameters
            simulator = st.session_state.simulator
            simulator.virtual_size = virtual_size
            simulator.page_size = page_size
            simulator.physical_size = physical_size
            simulator.conversion_type = conversion_type
            
            # Calculate memory parameters and initialize page table
            simulator.calculate_memory_parameters()
            
            # Update session state
            st.session_state.page_table_initialized = True
            st.session_state.address_results = []
            
            st.success("Memory configuration initialized successfully!")
        else:
            st.error(error_message)
    
    # Button to restart simulation
    if st.button("Restart Simulation"):
        restart_simulation()
        st.success("Simulation restarted!")

# Main content area
if not st.session_state.page_table_initialized:
    # Display instructions when not initialized
    st.info("Please configure memory parameters in the sidebar and click 'Initialize Memory' to start.")
    
    # Display diagram explaining memory paging
    st.subheader("Memory Paging Concept")
    st.write("""
    Memory paging is a memory management scheme that eliminates the need for contiguous allocation of physical memory.
    - Virtual addresses are divided into a page number and an offset
    - The page table maps virtual pages to physical frames
    - Page faults occur when a referenced page is not in physical memory
    """)
    
    # Create a simple diagram using markdown
    st.markdown("""
    ```
    Virtual Address Space              Physical Address Space
    +-------------------+              +-------------------+
    |                   |              |                   |
    |  Virtual Page 0   | ----+        |  Physical Frame 0 |
    |                   |     |        |                   |
    +-------------------+     |        +-------------------+
    |                   |     |        |                   |
    |  Virtual Page 1   | ----+------> |  Physical Frame 1 |
    |                   |     |        |                   |
    +-------------------+     |        +-------------------+
    |                   |     |        |                   |
    |  Virtual Page 2   | ----+        |  Physical Frame 2 |
    |                   |              |                   |
    +-------------------+              +-------------------+
    |                   |              |                   |
    |  Virtual Page 3   |              |  Physical Frame 3 |
    |                   |              |                   |
    +-------------------+              +-------------------+
    ```
    """)
else:
    # Display memory parameters
    simulator = st.session_state.simulator
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Virtual Size", f"{simulator.virtual_size} KB")
    with col2:
        st.metric("Physical Size", f"{simulator.physical_size} KB")
    with col3:
        st.metric("Page Size", f"{simulator.page_size} KB")
    
    # Manual Page Table Configuration Section
    st.subheader("Page Table Configuration")
    
    # Add feature to manually set page table entries
    with st.expander("Configure Page Table Entries and Arrival Order"):
        st.write("Set which virtual pages are present in physical memory and their corresponding physical frames.")
        
        # Flag to track if any changes were made to the page table
        page_table_changed = False
        
        # Get number of available physical frames
        num_physical_frames = simulator.num_physical_frames
        
        # Create a list of all potential physical frames
        available_frames = list(range(num_physical_frames))
        
        # Create columns for each page table property
        cols = st.columns([2, 1, 2, 2])
        cols[0].write("**Virtual Page**")
        cols[1].write("**Present**")
        cols[2].write("**Physical Frame**")
        cols[3].write("**Arrival Order**")
        
        # Used to track which frames are already assigned
        used_frames = set()
        arrival_order_list = []
        
        # For each virtual page, create a row of inputs
        for i, entry in enumerate(simulator.memory_map):
            cols = st.columns([2, 1, 2, 2])
            
            # Virtual page (display only)
            cols[0].write(f"`{entry['virtual_binary']}`")
            
            # Present bit (checkbox)
            present = cols[1].checkbox("", value=entry['present'], key=f"present_{i}")
            
            # Physical frame (selectbox, only enabled if present is checked)
            used_frame = entry['physical_index'] if entry['physical_index'] is not None else -1
            
            if present:
                # If this frame was already present, don't count it as "used" for other pages
                if used_frame >= 0:
                    frame_options = [used_frame] + [f for f in available_frames if f not in used_frames or f == used_frame]
                else:
                    frame_options = [f for f in available_frames if f not in used_frames]
                
                # If no frames available, select first one and display a warning
                if not frame_options:
                    frame_options = [0]
                    st.warning("⚠️ More pages selected as present than available physical frames!")
                
                physical_frame = cols[2].selectbox(
                    "", 
                    options=frame_options,
                    index=0 if used_frame not in frame_options else frame_options.index(used_frame),
                    key=f"frame_{i}"
                )
                
                # Mark this frame as used
                used_frames.add(physical_frame)
                
                # Add to arrival order list
                arrival_order_list.append((i, physical_frame))
            else:
                # Display disabled selectbox if not present
                cols[2].selectbox(
                    "", 
                    options=[-1], 
                    format_func=lambda x: "N/A",
                    disabled=True,
                    key=f"frame_disabled_{i}"
                )
                physical_frame = None
            
            # Check if page table entry changed
            if (present != entry['present']) or (present and physical_frame != entry['physical_index']):
                page_table_changed = True
                simulator.update_page_table(i, physical_frame, present)
        
        # Arrival order configuration
        st.write("---")
        st.write("**Configure FIFO Queue Arrival Order**")
        st.write("Drag to reorder the pages in the FIFO queue (oldest first):")
        
        # Get only the present pages
        present_pages = [(i, entry) for i, entry in enumerate(simulator.memory_map) if entry['present']]
        
        if present_pages:
            # Create a list of present pages to be displayed in arrival order editor
            arrival_options = []
            for i, entry in present_pages:
                virt_page = entry['virtual_binary']
                phys_frame = entry['physical_binary']
                arrival_options.append({"id": i, "label": f"Virtual Page {virt_page} → Physical Frame {phys_frame}"})
            
            if 'fifo_order' not in st.session_state:
                # Initialize with current FIFO queue if it exists
                if simulator.fifo_queue:
                    st.session_state.fifo_order = list(simulator.fifo_queue)
                else:
                    # Initialize with pages in numerical order
                    st.session_state.fifo_order = [entry["id"] for entry in arrival_options]
            
            # Reorder if necessary to show only present pages and remove non-present ones
            present_ids = [entry["id"] for entry in arrival_options]
            st.session_state.fifo_order = [id for id in st.session_state.fifo_order if id in present_ids]
            for id in present_ids:
                if id not in st.session_state.fifo_order:
                    st.session_state.fifo_order.append(id)
            
            # Display the current order
            fifo_editor_cols = st.columns(len(st.session_state.fifo_order))
            for i, page_id in enumerate(st.session_state.fifo_order):
                # Find the label for this page
                label = next((entry["label"] for entry in arrival_options if entry["id"] == page_id), f"Page {page_id}")
                fifo_editor_cols[i].write(f"{i+1}. {label}")
            
            # Create a multi-select for reordering
            new_order = st.multiselect(
                "Select pages in desired arrival order (oldest first):",
                options=arrival_options,
                default=arrival_options,
                format_func=lambda x: x["label"],
                key="fifo_reorder"
            )
            
            # Update FIFO queue if order changed
            if st.button("Update Arrival Order"):
                if new_order:
                    new_order_ids = [item["id"] for item in new_order]
                    if new_order_ids != st.session_state.fifo_order:
                        # Update the FIFO queue
                        simulator.fifo_queue = deque(new_order_ids)
                        st.session_state.fifo_order = new_order_ids
                        st.success("FIFO queue order updated!")
                        page_table_changed = True
                else:
                    st.error("Please select at least one page for the FIFO queue")
        else:
            st.info("No pages are currently present in physical memory. Add pages to configure arrival order.")
    
    # Display page table
    st.subheader("Page Table")
    
    # Create a DataFrame for the page table
    page_table_data = []
    for entry in simulator.memory_map:
        in_queue = entry['virtual_index'] in simulator.fifo_queue
        queue_pos = list(simulator.fifo_queue).index(entry['virtual_index']) + 1 if in_queue else None
        
        page_table_data.append({
            "Virtual Page": entry['virtual_binary'],
            "Physical Frame": entry['physical_binary'] if entry['present'] else "Not Present",
            "Present Bit": "1" if entry['present'] else "0",
            "In FIFO Queue": f"Yes (Position {queue_pos})" if in_queue else "No"
        })
    
    page_table_df = pd.DataFrame(page_table_data)
    st.dataframe(page_table_df, use_container_width=True)
    
    # Display FIFO queue
    if simulator.fifo_queue:
        st.subheader("FIFO Page Replacement Queue")
        st.write("Pages in order of arrival (oldest first):")
        fifo_display = [f"Page {dec_to_bin(simulator.index_bits, page)}" for page in simulator.fifo_queue]
        st.code(" → ".join(fifo_display))
    
    # Form for address conversion
    st.subheader("Address Conversion")
    
    with st.form("address_conversion_form"):
        if simulator.conversion_type == "virtual_to_phys":
            address_label = "Virtual Address"
            convert_button_label = "Convert to Physical Address"
        else:
            address_label = "Physical Address"
            convert_button_label = "Convert to Virtual Address"
        
        address_input = st.text_input(address_label, help="Enter a decimal number")
        convert_submitted = st.form_submit_button(convert_button_label)
        
        if convert_submitted and address_input:
            try:
                address = int(address_input)
                max_address = 2**simulator.total_bits - 1
                
                if address < 0 or address > max_address:
                    st.error(f"Address must be between 0 and {max_address}")
                else:
                    result = simulator.convert_address(address)
                    
                    if result['result_status'] == 'success':
                        # Add the result to the history
                        st.session_state.address_results.append(result)
                        
                        # Display conversion information
                        st.success("Address conversion successful!")
                        
                        st.markdown("### Conversion Details")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Original Address:**")
                            st.code(f"Decimal: {result['original_address_dec']}")
                            st.code(f"Binary: {result['original_address_bin']}")
                            
                            st.markdown(f"**Page Index:**")
                            st.code(f"Binary: {result['page_index_bin']}")
                            st.code(f"Decimal: {result['page_index_dec']}")
                            
                            st.markdown(f"**Offset:**")
                            st.code(f"Binary: {result['offset_bin']}")
                            st.code(f"Decimal: {result['offset_dec']}")
                        
                        with col2:
                            st.markdown(f"**Converted Address:**")
                            st.code(f"Decimal: {result['converted_address_dec']}")
                            st.code(f"Binary: {result['converted_address_bin']}")
                            
                            if result['page_fault']:
                                st.warning("⚠️ Page fault occurred!")
                                if result['page_replaced'] is not None:
                                    st.info(f"Page {result['page_replaced']} was replaced using FIFO")
                    else:
                        st.error(f"Error in conversion: {result.get('error_message', 'Unknown error')}")
            except ValueError:
                st.error("Please enter a valid decimal number")
    
    # Display conversion history
    if st.session_state.address_results:
        st.subheader("Conversion History")
        
        # Display most recent conversions first
        for i, result in enumerate(reversed(st.session_state.address_results[-5:])):
            with st.expander(f"Conversion #{len(st.session_state.address_results) - i}"):
                if simulator.conversion_type == "virtual_to_phys":
                    st.write(f"Virtual Address {result['original_address_dec']} → Physical Address {result['converted_address_dec']}")
                else:
                    st.write(f"Physical Address {result['original_address_dec']} → Virtual Address {result['converted_address_dec']}")
                
                if result['page_fault']:
                    st.write("⚠️ Page fault occurred!")
                    if result['page_replaced'] is not None:
                        st.write(f"Page {result['page_replaced']} was replaced using FIFO")
    
