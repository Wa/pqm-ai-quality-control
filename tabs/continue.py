if st.session_state.get(f"enterprise_continue_running_{session_id}"):
	# Retrieve context and init dirs
	std_txt_files = st.session_state.get(f"enterprise_std_txt_files_{session_id}") or []
	exam_txt_files = st.session_state.get(f"enterprise_exam_txt_files_{session_id}") or []
	enterprise_out = st.session_state.get(f"enterprise_out_root_{session_id}") or enterprise_out_root
	initial_dir = os.path.join(enterprise_out, 'initial_results')
	os.makedirs(initial_dir, exist_ok=True)
	checkpoint_dir = os.path.join(enterprise_out, 'checkpoint')
	manifest_path = os.path.join(checkpoint_dir, 'manifest.json')
	_manifest = None
	try:
		import json as _json
		with open(manifest_path, 'r', encoding='utf-8') as _mf:
			_manifest = _json.load(_mf) or {}
	except Exception:
		_manifest = None
	if not _manifest or not isinstance(_manifest.get('entries'), list) or not _manifest['entries']:
		st.info("未发现跑到一半的项目")
		st.session_state[f"enterprise_continue_running_{session_id}"] = False
	else:
		# group by file in entry order
		from collections import OrderedDict as _OD
		_entries_sorted = sorted((_manifest.get('entries') or []), key=lambda e: int(e.get('id', 0)))
		_groups = _OD()
		for _e in _entries_sorted:
			fname = str(_e.get('file_name', ''))
			_groups.setdefault(fname, []).append(_e)
		bisheng_session_id = st.session_state.get(f"bisheng_session_{session_id}")
		for name, _elist in _groups.items():
			st.markdown(f"**📄 继续比对：{name}**")
			full_out_text = ""
			prompt_texts = []
			_total_parts = len(_elist)
			try:
				_elist.sort(key=lambda e: (int(e.get('chunk_index', 0)), int(e.get('id', 0))))
			except Exception:
				pass
			for _i, _entry in enumerate(_elist, start=1):
				_prompt_file = str(_entry.get('prompt_file', ''))
				_response_file = str(_entry.get('response_file', ''))
				_status = str(_entry.get('status', 'not_done'))
				_prompt_path = os.path.join(checkpoint_dir, _prompt_file)
				_response_path = os.path.join(checkpoint_dir, _response_file)
				# load prompt
				try:
					with open(_prompt_path, 'r', encoding='utf-8') as _pf:
						prompt_text = _pf.read()
				except Exception:
					prompt_text = ""
				prompt_texts.append(prompt_text)
				col_prompt, col_response = st.columns([1, 1])
				with col_prompt:
					st.markdown(f"提示词（第{_i}部分，共{_total_parts}部分）")
					prompt_container = st.container(height=400)
					with prompt_container:
						with st.chat_message("user"):
							prompt_placeholder = st.empty()
							words = (prompt_text or "").split()
							streamed = ""
							for j in range(0, len(words), 30):
								chunk_words = words[j:j+30]
								streamed += " ".join(chunk_words) + " "
								prompt_placeholder.text(streamed.strip())
					st.chat_input(placeholder="", disabled=True, key=f"enterprise_continue_prompt_{session_id}_{name}_{_i}")
				with col_response:
					st.markdown(f"AI比对结果（第{_i}部分，共{_total_parts}部分）")
					response_container = st.container(height=400)
					with response_container:
						with st.chat_message("assistant"):
							ph = st.empty()
							if _status == 'done':
								# playback response
								try:
									with open(_response_path, 'r', encoding='utf-8') as _rf:
										_resp_text = _rf.read()
								except Exception:
									_resp_text = ""
								acc = ""
								words_r = (_resp_text or "").split()
								for k in range(0, len(words_r), 30):
									chunk_words = words_r[k:k+30]
									acc += " ".join(chunk_words) + " "
									ph.write(acc.strip())
									import time as _t
									_t.sleep(0.1)
								full_out_text += ("\n\n" if full_out_text else "") + (_resp_text or "")
							else:
								# run LLM and update manifest
								try:
									kb_name_dyn = f"{session_id}_{TAB_SLUG}"
									kid = find_knowledge_id_by_name(BISHENG_BASE_URL, BISHENG_API_KEY or None, kb_name_dyn)
									response_text = ""
									res = call_flow_process(
										base_url=BISHENG_BASE_URL,
										flow_id=BISHENG_FLOW_ID,
										question=prompt_text,
										kb_id=kid,
										input_node_id=FLOW_INPUT_NODE_ID,
										api_key=BISHENG_API_KEY or None,
										session_id=bisheng_session_id,
										history_count=0,
										extra_tweaks=None,
										milvus_node_id=FLOW_MILVUS_NODE_ID,
										es_node_id=FLOW_ES_NODE_ID,
										timeout_s=180,
										max_retries=2,
									)
									ans_text, new_sid = parse_flow_answer(res)
									if new_sid:
										bisheng_session_id = new_sid
										st.session_state[f"bisheng_session_{session_id}"] = bisheng_session_id
									ph.write(ans_text or "")
									full_out_text += ("\n\n" if full_out_text else "") + (ans_text or "")
									# save resp and mark done
									try:
										with open(_response_path, 'w', encoding='utf-8') as _rf:
											_rf.write(ans_text or "")
									except Exception:
										pass
									try:
										import json as _json, tempfile as _tmp, shutil as _sh
										for __e in _manifest.get('entries', []):
											if int(__e.get('id', 0)) == int(_entry.get('id', 0)):
												__e['status'] = 'done'
												break
										with _tmp.NamedTemporaryFile('w', delete=False, encoding='utf-8', dir=checkpoint_dir) as _tf:
											_tf.write(_json.dumps(_manifest, ensure_ascii=False, indent=2))
											_tmpname = _tf.name
										_sh.move(_tmpname, manifest_path)
									except Exception:
										pass
								except Exception as e:
									ph.error(f"调用失败：{e}")
					st.chat_input(placeholder="", disabled=True, key=f"enterprise_continue_response_{session_id}_{name}_{_i}")
			# write combined prompt/response for this file
			name_no_ext = os.path.splitext(name)[0]
			try:
				# prompts
				total_parts = len(prompt_texts)
				prompt_out_lines = []
				for idx_p, ptxt in enumerate(prompt_texts, start=1):
					prompt_out_lines.append(f"提示词（第{idx_p}部分，共{total_parts}部分）：")
					prompt_out_lines.append(ptxt)
				prompt_out_text = "\n".join(prompt_out_lines)
				with open(os.path.join(initial_dir, f"prompt_{name_no_ext}.txt"), 'w', encoding='utf-8') as pf:
					pf.write(prompt_out_text)
			except Exception:
				pass
			try:
				with open(os.path.join(initial_dir, f"response_{name_no_ext}.txt"), 'w', encoding='utf-8') as outf:
					outf.write(full_out_text)
			except Exception:
				pass
	# End continue branch
# The end