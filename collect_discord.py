from datetime import datetime, timedelta
import discord

# Discord bot token
TOKEN = 'token'

CHANNEL_ID = 655899513965510681 

intents = discord.Intents.default()
intents.messages = True 
intents.guilds = True  

# Define your bot client
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

    channel = client.get_channel(CHANNEL_ID)
    
    if channel is None:
        print("Channel not found.")
        return
    today = datetime.now()

    # Calculate how many days until next Sunday
    days_until_sunday = (6 - today.weekday()) % 7

    if days_until_sunday == 0:
        days_until_sunday = 7

    next_sunday = today + timedelta(days=days_until_sunday)
    # How far you go back (should change to automatic)
    number_of_weeks = 208
    one_week = timedelta(days=7)

    weekly_message_counts = {}

    for i in range(number_of_weeks):
        end_of_week = next_sunday - (i * one_week)
        start_of_week = end_of_week - one_week
        
        message_count = 0
        async for message in channel.history(after=start_of_week, before=end_of_week):
            message_count += 1
        
        weekly_message_counts[f"Week {i + 1}"] = message_count
        print(f"Messages for Week {i + 1} (from {start_of_week} to {end_of_week}): {message_count}")

    print("\nFinal Message Counts:")
    for week, count in weekly_message_counts.items():
        print(f"{week} : {count}")

    await client.close()

client.run(TOKEN)
