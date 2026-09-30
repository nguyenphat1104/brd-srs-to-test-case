# Source review notes for the two-document pilot

These notes were prepared from the source PDFs before any condition outputs were generated. They are assistant-authored review aids, not independent human ratings or approval.

## D001 — Model Manager

Check that the catalog preserves:

- Stand-alone and MetVault-connected use, web GUI and command-line access.
- Automatic cluster selection and the user's explicit cluster choice; multi-cluster ensembles.
- Distinct Weather FDDA, ClimoFDDA, post-processing, by-hand and saved/file configuration workflows. CAM setup is explicitly TBD, not an implemented workflow with invented steps.
- MM5 versus WRF choices, real-time versus off-line jobs, and the past-cycle prompt distinguishing a case study from a re-run.
- Standard/custom input data and processors. For a re-run, MetVault returns the data available and used in that historical cycle, not automatically the newest data.
- Optional post-processing versus “submit now” behavior without post-processing; configuration saving and resubmission.
- Mandatory versus optional by-hand job information. Preserve the source's uncertainty around number of nodes; do not invent mandatory values or rejection messages.
- Four job views: running, scheduled, past and all. Job logs and state-specific queue/running/stopped operations are distinct.
- Ordinary users act on their own jobs; a super user can delete/stop/restart/resume any applicable job. Do not collapse these actors.

Record ambiguities rather than supply defaults: domain creation and WRF applicability; restart-file frequency; unspecified WRF options; Final/Preliminary Analysis applicability; exact additional processing choices; MEDOC ranges marked with a question; coupled-app behavior; job priority; and unspecified detail-view fields. Some displayed job-table attributes are introduced with “may”, so the catalog must not silently convert every possible column into an unconditional requirement.

## D002 — DigitalHome 1.3

Check exact numbers, inclusivity, units, actors and timing:

| Source location | Constraint or distinction |
|---|---|
| PDF page 7, §3.4.4.1(c) | Sensor sensitivity: 14–104°F (−10–40°C). This differs from the user's narrower thermostat setting range. |
| PDF page 9, §4.2.1.2 | Thermostat setting: 60–80°F inclusive, increments of 1°F. |
| PDF page 9, §4.2.2 | At most eight thermostats; individual or collective control; one thermostat per controlled enclosed space. |
| PDF page 9, §4.2.2.3 | Up to 24 one-hour thermostat settings per day, for every day of the week. |
| PDF page 9, §4.3.1.2 | Humidity setting: 30–60% inclusive, increments of 1%. |
| PDF pages 9–10, §4.3.2 | At most eight humidistats; one per controlled enclosed space; up to 24 one-hour settings per day. |
| PDF pages 9–10 | Manual overrides persist until the current planned/default period ends; the next period uses its planned/default setting. |
| PDF page 10, §4.4 | Up to 50 contact sensors; sound and light alarm subsystems; alarm activation on the specified breach/contact condition. |
| PDF page 10, §4.5 | Up to 100 power switches, rated 115 V / 10 A; read ON/OFF and change either direction. |
| PDF pages 10–11, §4.6.2.1 | Monthly planner: up to four daily time periods. Do not replace this with the separate 24-hour device schedule limit. |
| PDF page 11, §4.6.3 | Monthly reports available for the preceding two years, with daily averages/extrema and their times, security events and downtime. |
| PDF page 11, §5.1 | Display update interval no longer than two seconds; sensor acquisition at least 10 Hz; communication within 1,000 feet. |
| PDF page 11, §5.2 | No more than one failure per 10,000 operating hours; daily backups at technician-configured time; recovery from the most recent backup. |
| PDF pages 5–6, 12 | Distinguish general user, master user and technician privileges. Authentication requires account name/password; protect information in transit using the specified class of security technology. |
| PDF pages 6, 12–13 | Simulated environment constraints, error reporting, maintenance/documentation obligations and user help remain in scope where testable by inspection. |

Source issues requiring an explicit reviewer decision:

1. §4.3.2.2 says a humidistat allows a manual **temperature** setting. It may be a source typo; retain an ambiguity note rather than silently claiming corrected source text.
2. The 24-hour device schedules and four-period monthly planner are separate statements; their interaction is not fully specified.
3. §4.4.2 and the glossary describe contact state/security activation with different wording. Do not invent missing arming/authorization semantics.
4. The source cites external standards and a supplementary use-case document that are not supplied here. Do not invent their detailed requirements or test oracles.
5. Project staffing, schedules and organizational approvals should be distinguished from runtime product behavior. If included as process/inspection coverage units, classify them explicitly and apply the same scope across all conditions.

## Human review response

After reading the catalogs, identify missing, duplicate or unsupported units by document and unit ID. Suggested response: “D001 CU-___ needs ___; D002 CU-___ needs ___.” Approve only the resulting source catalogs, before viewing generated condition suites. This is separate from the later blinded two-rater evaluation.
