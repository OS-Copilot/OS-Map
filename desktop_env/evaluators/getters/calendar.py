import os
import re
# TODO: pip install ics
from ics import Calendar

from desktop_env.evaluators.getters.file import get_vm_file


def get_calendar_events(env, config):
    ics_file_vm_path = "/home/user/.local/share/evolution/calendar/system/calendar.ics"
    ics_file_local_path = get_vm_file(env, {"path": ics_file_vm_path, "dest": os.path.split(ics_file_vm_path)[-1]})

    
    with open(ics_file_local_path, 'r', encoding='utf-8') as f:
        calendar = Calendar(f.read())
    
    events = []
    for event in calendar.events:
        event_dict = {"from": event.begin.strftime("%Y%m%dT%H%M%S"), "to": event.end.strftime("%Y%m%dT%H%M%S"), "title": event.name}
        if event.location:
            event_dict["location"] = event.location
        events.append(event_dict)
    return events
