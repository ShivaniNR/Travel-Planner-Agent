import os
from dotenv import load_dotenv
import streamlit as st
import requests
from langchain_tavily import TavilySearch
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
import time

# Setup streamlit app
st.title('AI Travel Planner')

# Loading environment variables
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# System resource monitoring
def monitor_system():
    """Basic system monitoring"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        return cpu_percent, memory_percent
    except ImportError:
        return None, None


search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# OPTIMIZED PROMPTS FOR GEMINI
researcher_agent_prompt = PromptTemplate(
    template="""You are an efficient travel researcher using Google Gemini. Research key travel information quickly.

Available tools: {tools}
Tool names: [{tool_names}]

EFFICIENT RESEARCH STRATEGY (4-6 searches max):
1. Search top attractions and activities
2. Search accommodations and neighborhoods  
3. Search restaurants and local cuisine
4. Search transportation and practical tips

Follow this format exactly:
Question: the input question you must answer
Thought: what I need to find out
Action: the action to take, should be one of [{tool_names}]
Action Input: the search query
Observation: the result of the action
... (repeat if needed, but limit to 6 iterations)
Thought: I now have sufficient information
Final Answer: comprehensive travel guide summary

Question: {input}
Thought:{agent_scratchpad}""",
    input_variables=["input", "agent_scratchpad"],
    partial_variables={
        "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        "tool_names": ", ".join([tool.name for tool in tools])
    }
)

planner_agent_prompt = PromptTemplate(
    template="""You are a travel planner using Google Gemini. Create detailed itineraries efficiently.

Available tools: {tools}
Tool names: [{tool_names}]

EFFICIENT PLANNING (use 2-3 searches max if needed):
- Use the provided research data primarily
- Only search if critical information is missing
- Focus on creating a practical, day-by-day itinerary

Follow this format:
Question: the input question you must answer
Thought: what I need to do
Action: the action to take, should be one of [{tool_names}]
Action Input: [search query if needed]
Observation: [result if searched]
Thought: I can create the itinerary now
Final Answer: [complete day-by-day itinerary with timing, activities, restaurants, and tips]

Question: {input}
Thought:{agent_scratchpad}""",
    input_variables=["input", "agent_scratchpad"],
    partial_variables={
        "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        "tool_names": ", ".join([tool.name for tool in tools])
    }
)

# GEMINI LLM CONFIGURATION
@st.cache_resource
def get_gemini_llm():
    """Cached Gemini LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=os.environ.get('GOOGLE_API_KEY')
    )

def cooling_break(seconds=1):
    """Minimal break"""
    time.sleep(seconds)

# Check API keys
if not os.environ.get('TAVILY_API_KEY'):
    st.error("🔑 Tavily API key missing! Add TAVILY_API_KEY to your .env file")
    st.info("Get your free Tavily API key at: https://tavily.com")
    st.stop()

if not os.environ.get('GOOGLE_API_KEY'):
    st.error("🔑 Google API key missing! Add GOOGLE_API_KEY to your .env file")
    st.info("Get your free Gemini API key at: https://makersuite.google.com/app/apikey")
    st.stop()

# Initialize components
try:
    llm = get_gemini_llm()
    tools = [search_tool]
    
    # Get tool descriptions
    tools_desc = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    
    # FIXED EXECUTORS with all required variables
    researcher_agent = create_react_agent(
        llm=llm, 
        tools=tools, 
        prompt=researcher_agent_prompt.partial(
            tools=tools_desc, 
            tool_names=tool_names
        )
    )
    
    researcher_executor = AgentExecutor(
        agent=researcher_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        max_execution_time=120,
        return_intermediate_steps=False
    )
    
    planner_agent = create_react_agent(
        llm=llm, 
        tools=tools, 
        prompt=planner_agent_prompt.partial(
            tools=tools_desc, 
            tool_names=tool_names
        )
    )
    
    planner_executor = AgentExecutor(
        agent=planner_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        max_execution_time=120,
        return_intermediate_steps=False
    )
    
    
except Exception as e:
    st.error(f"❌ Failed to initialize: {str(e)}")
    st.info("Check your API keys in the .env file")
    
    # Debug information
    with st.expander("🔧 Debug Information"):
        st.write("Error details:", str(e))
        st.write("Google API Key set:", bool(os.environ.get('GOOGLE_API_KEY')))
        st.write("Tavily API Key set:", bool(os.environ.get('TAVILY_API_KEY')))
        st.write("Search tool type:", type(search_tool).__name__)
        st.write("Search tool name:", getattr(search_tool, 'name', 'No name attribute'))
    st.stop()


# System status
# col1, col2, col3 = st.columns(3)
# with col1:
#     if st.button("🌡️ System Check"):
#         cpu_usage, memory_usage = monitor_system()
#         if cpu_usage and memory_usage:
#             if cpu_usage < 50:
#                 st.success(f"✅ CPU: {cpu_usage}%")
#             else:
#                 st.warning(f"⚠️ CPU: {cpu_usage}%")
#             st.info(f"💾 Memory: {memory_usage}%")
#         else:
#             st.info("Install 'psutil' for system monitoring")

# with col2:
#     if st.button("🧪 Test Gemini"):
#         try:
#             test_response = llm.invoke("Say 'Gemini working!' in one sentence")
#             st.success(f"✅ {test_response.content}")
#         except Exception as e:
#             st.error(f"❌ Gemini test failed: {str(e)}")

# with col3:
#     emergency_stop = st.button("🛑 STOP", type="primary")
#     if emergency_stop:
#         st.stop()

# Input fields
destination = st.text_input("🗺️ Destination:", placeholder="e.g., Tokyo, Paris, New York")
num_days = st.number_input("📅 Number of days:", min_value=1, max_value=21, value=5)

# Preferences
with st.expander("🎯 Preferences (Optional)"):
    col1, col2 = st.columns(2)
    with col1:
        budget = st.selectbox("💰 Budget:", ["Budget-friendly", "Mid-range", "Luxury", "No preference"])
        travel_style = st.selectbox("🚶 Travel style:", ["Relaxed", "Moderate", "Fast-paced"])
    with col2:
        interests = st.multiselect("🎨 Interests:", 
                                 ["History & Culture", "Food & Dining", "Nature & Outdoors", 
                                  "Museums & Art", "Nightlife", "Shopping", "Adventure Sports"])
        group_size = st.selectbox("👥 Group size:", ["Solo", "Couple", "Family", "Group"])

# Initialize session state for results persistence
if 'itinerary_results' not in st.session_state:
    st.session_state.itinerary_results = None
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
# if 'generation_time' not in st.session_state:
#     st.session_state.generation_time = None
if 'last_destination' not in st.session_state:
    st.session_state.last_destination = None
if 'last_preferences' not in st.session_state:
    st.session_state.last_preferences = None
if 'has_previous_results' not in st.session_state:
    st.session_state.has_previous_results = False

# Main execution
if st.button("🚀 Generate Itinerary", type="primary"):
    if destination and num_days:
        try:            
            # PHASE 1: RESEARCH - Simplified query
            with st.spinner("🔍 Researching..."):
                research_query = f"Research {destination} travel guide for {num_days} days including top attractions, restaurants, transportation, accommodation areas, and practical tips"
                
                st.info("🔍 Starting research phase...")
                research_results = researcher_executor.invoke({"input": research_query})
                cooling_break(2)
                
            # Check if research was successful
            research_output = research_results.get('output', '')
            if "Agent stopped due to iteration limit" in research_output or len(research_output.strip()) < 50:
                st.warning("⚠️ Research phase incomplete. Using fallback approach...")
                
                # Fallback: Direct search
                try:
                    fallback_search = search_tool.invoke(f"{destination} travel attractions restaurants {num_days} days")
                    if isinstance(fallback_search, list):
                        research_output = "\n".join([str(item) for item in fallback_search[:3]])
                    else:
                        research_output = str(fallback_search)
                    st.info("✅ Fallback research completed")
                except Exception as e:
                    research_output = f"Basic travel information for {destination}. Popular destination with many attractions, restaurants, and accommodation options."
                    st.warning(f"⚠️ Fallback also failed: {str(e)}")
            else:
                st.success("✅ Research completed!")
            
            
            # PHASE 2: PLANNING  
            with st.spinner("📋 Heading to create your itinerary..."):
                planning_query = f"""
Create a detailed {num_days}-day itinerary for {destination} based on this research:

=== RESEARCH DATA ===
{research_output}

=== USER PREFERENCES ===
• Budget: {budget}
• Travel Style: {travel_style}
• Interests: {', '.join(interests) if interests else 'General sightseeing'}
• Group: {group_size}
• Duration: {num_days} days

Create a practical day-by-day itinerary with specific timing, activities, restaurants, transportation tips, and estimated costs.
                """
                
                st.info("📋 Starting planning phase...")
                itinerary_results = planner_executor.invoke({"input": planning_query})
                cooling_break(1)
            
            
            # Store results in session state to persist across reruns
            st.session_state.itinerary_results = itinerary_results.get('output', '')
            st.session_state.research_results = research_output
            st.session_state.last_destination = destination
            st.session_state.last_preferences = {
                'budget': budget,
                'travel_style': travel_style,
                'interests': interests,
                'group_size': group_size,
                'num_days': num_days
            }
            
            # DISPLAY RESULTS
            st.markdown("## 🗺️ Your Personalized Travel Itinerary")
            
            # Check if planning was successful
            if "Agent stopped due to iteration limit" in st.session_state.itinerary_results or len(st.session_state.itinerary_results.strip()) < 100:
                st.error("❌ Itinerary generation failed. Please try again with a simpler destination.")
                with st.expander("🔧 Debug Information"):
                    st.text("Research Output Length: " + str(len(st.session_state.research_results)))
                    st.text("Itinerary Output: " + st.session_state.itinerary_results[:200] + "...")
            else:
                st.markdown(st.session_state.itinerary_results)
                
                # Download options
                itinerary_text = f"""
                        {st.session_state.last_destination.upper()} - {st.session_state.last_preferences['num_days']} DAY ITINERARY
                        Budget: {st.session_state.last_preferences['budget']} | Style: {st.session_state.last_preferences['travel_style']} | Group: {st.session_state.last_preferences['group_size']}
                        {'='*50}

                        {st.session_state.itinerary_results}

                        {'='*50}
                        Research Data:
                        {st.session_state.research_results}
                        """
                st.download_button(
                    label="📄 Download Complete Itinerary",
                    data=itinerary_text,
                    file_name=f"{st.session_state.last_destination.replace(' ', '_')}_{st.session_state.last_preferences['num_days']}days.txt",
                    mime="text/plain",
                    key="download_complete"  # Unique key to prevent conflicts
                )
                # col1, col2 = st.columns(2)
                
                # with col1:
                    
                
                # with col2:
                #     # Quick summary
                #     summary = f"{st.session_state.last_destination} • {st.session_state.last_preferences['num_days']} days • {st.session_state.last_preferences['budget']} • Generated: {time.strftime('%Y-%m-%d %H:%M')}"
                #     st.download_button(
                #         label="📋 Download Summary",
                #         data=f"TRIP SUMMARY\n{summary}\n\n{st.session_state.itinerary_results}",
                #         file_name=f"{st.session_state.last_destination.replace(' ', '_')}_summary.txt",
                #         mime="text/plain",
                #         key="download_summary"  # Unique key to prevent conflicts
                #     )
            
            # Research details
            with st.expander("📊 View Detailed Research"):
                st.markdown(st.session_state.research_results[:2000] + "..." if len(st.session_state.research_results) > 2000 else st.session_state.research_results)
            
            # Feedback section
            st.markdown("---")
            with st.container():
                st.markdown("### 💭 Rate your experience:")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("😍 Excellent"):
                        st.balloons()
                        st.success("Thank you! Gemini rocks! 🚀")
                with col2:
                    if st.button("👍 Good"):
                        st.success("Thanks for the feedback!")
                with col3:
                    if st.button("😐 Okay"):
                        st.info("We'll keep improving!")
                with col4:
                    if st.button("👎 Poor"):
                        st.info("Sorry! Please try different preferences.")
                        
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            
            # Enhanced debug information
            with st.expander("🔧 Debug Information"):
                st.write("Error details:", str(e))
                st.write("Error type:", type(e).__name__)
                st.write("Destination:", destination)
                st.write("Days:", num_days)
                st.write("Google API Key set:", bool(os.environ.get('GOOGLE_API_KEY')))
                st.write("Tavily API Key set:", bool(os.environ.get('TAVILY_API_KEY')))
                
                # Test API keys
                try:
                    test_llm = get_gemini_llm()
                    test_response = test_llm.invoke("test")
                    st.write("Gemini test successful:", True)
                except Exception as gemini_error:
                    st.write("Gemini test error:", str(gemini_error))
                
                try:
                    test_search = search_tool.invoke("test query")
                    st.write("Tavily test successful:", True)
                except Exception as tavily_error:
                    st.write("Tavily test error:", str(tavily_error))
            
            st.info("""
            💡 **Troubleshooting:**
            - Check your Google API key is valid
            - Verify Tavily API key is correct
            - Try a simpler destination name (e.g., "Paris" instead of "Paris, France")
            - Reduce number of days to 3-4
            - Check internet connection
            - Clear browser cache and refresh
            """)
    else:
        st.warning("⚠️ Please enter both destination and number of days.")

# # Display previously generated results if they exist (after download button clicks)
# if st.session_state.last_destination:
#     st.markdown("---")
#     st.markdown("## 📋 Previously Generated Itinerary")
#     st.info(f"🗺️ **{st.session_state.last_destination}** • {st.session_state.last_preferences['num_days']} days • Generated with Google Gemini")
    
#     # Show results
#     with st.container():
#         st.markdown("### 🗺️ Your Travel Itinerary")
#         st.markdown(st.session_state.itinerary_results)
        
#         # Persistent download buttons (always available)
#         col1, col2 = st.columns(2)
        
#         with col1:
#             itinerary_text = f"""
#                 {st.session_state.last_destination.upper()} - {st.session_state.last_preferences['num_days']} DAY ITINERARY
#                 Budget: {st.session_state.last_preferences['budget']} | Style: {st.session_state.last_preferences['travel_style']} | Group: {st.session_state.last_preferences['group_size']}
#                 {'='*50}

#                 {st.session_state.itinerary_results}

#                 {'='*50}
#                 Research Data:
#                 {st.session_state.research_results}
#                 """
#             st.download_button(
#                 label="📄 Download Complete Itinerary",
#                 data=itinerary_text,
#                 file_name=f"{st.session_state.last_destination.replace(' ', '_')}_{st.session_state.last_preferences['num_days']}days.txt",
#                 mime="text/plain",
#                 key="persistent_download_complete"
#             )
        
        
#         with col2:
#             if st.button("🗑️ Clear Results", key="clear_results"):
#                 st.session_state.itinerary_results = None
#                 st.session_state.research_results = None
#                 st.session_state.generation_time = None
#                 st.session_state.last_destination = None
#                 st.session_state.last_preferences = None
#                 st.rerun()
        
#         # Research details (persistent)
#         with st.expander("📊 View Detailed Research"):
#             st.markdown(st.session_state.research_results[:2000] + "..." if len(st.session_state.research_results) > 2000 else st.session_state.research_results)