<script lang="ts">
	import { onMount, tick, getContext } from 'svelte';

	import Textarea from '$lib/components/common/Textarea.svelte';
	import { toast } from 'svelte-sonner';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { slugify } from '$lib/utils';
	import Spinner from '$lib/components/common/Spinner.svelte';

	export let onSubmit: Function;
	export let edit = false;
	export let prompt = null;
	export let clone = false;

	const i18n = getContext('i18n');

	let loading = false;

	let title = '';
	let command = '';
	let content = '';

	let hasManualEdit = false;

	$: if (!edit && !hasManualEdit) {
		command = title !== '' ? slugify(title) : '';
	}

	// Track manual edits
	function handleCommandInput(e: Event) {
		hasManualEdit = true;
	}

	const submitHandler = async () => {
		loading = true;

		if (validateCommandString(command)) {
			await onSubmit({
				title,
				command,
				content,
				access_control: {}
			});
		} else {
			toast.error(
				$i18n.t('Only alphanumeric characters and hyphens are allowed in the command string.')
			);
		}

		loading = false;
	};

	const validateCommandString = (inputString) => {
		// Regular expression to match only alphanumeric characters, hyphen, and underscore
		const regex = /^[a-zA-Z0-9-_]+$/;

		// Test the input string against the regular expression
		return regex.test(inputString);
	};

	onMount(async () => {
		if (prompt) {
			title = prompt.title;
			await tick();

			command = prompt.command.at(0) === '/' ? prompt.command.slice(1) : prompt.command;
			content = prompt.content;
		}
	});
</script>

<div class="w-full max-h-full flex justify-center">
	<form
		class="flex flex-col w-full mb-10"
		on:submit|preventDefault={() => {
			submitHandler();
		}}
	>
		<div class="my-2">
			<Tooltip
				content={`${$i18n.t('Only alphanumeric characters and hyphens are allowed')} - ${$i18n.t(
					'Activate this command by typing "/{{COMMAND}}" to chat input.',
					{
						COMMAND: command
					}
				)}`}
				placement="bottom-start"
			>
				<div class="flex flex-col w-full">
					<div class="flex items-center">
						<input
							class="text-2xl font-medium w-full bg-transparent outline-hidden"
							placeholder={$i18n.t('Title')}
							bind:value={title}
							required
						/>
					</div>

					<div class="flex gap-0.5 items-center text-xs text-gray-500">
						<div class="">/</div>
						<input
							class=" w-full bg-transparent outline-hidden"
							placeholder={$i18n.t('Command')}
							bind:value={command}
							on:input={handleCommandInput}
							required
							disabled={edit}
						/>
					</div>
				</div>
			</Tooltip>
		</div>

		<div class="my-2">
			<div class="flex w-full justify-between">
				<div class=" self-center text-sm font-semibold">{$i18n.t('Prompt Content')}</div>
			</div>

			<div class="mt-2">
				<div>
					<Textarea
						className="text-sm w-full bg-transparent outline-hidden overflow-y-hidden resize-none"
						placeholder={$i18n.t('Write a summary in 50 words that summarizes {{topic}}.')}
						bind:value={content}
						rows={6}
						required
					/>
				</div>

				<div class="text-xs text-gray-400 dark:text-gray-500">
					ⓘ {$i18n.t('Format your variables using brackets like this:')}&nbsp;<span
						class=" text-gray-600 dark:text-gray-300 font-medium"
						>{'{{'}{$i18n.t('variable')}{'}}'}</span
					>.
					{$i18n.t('Make sure to enclose them with')}
					<span class=" text-gray-600 dark:text-gray-300 font-medium">{'{{'}</span>
					{$i18n.t('and')}
					<span class=" text-gray-600 dark:text-gray-300 font-medium">{'}}'}</span>.
				</div>

				<div class="text-xs text-gray-400 dark:text-gray-500 underline">
					<a href="https://docs.openwebui.com/features/workspace/prompts" target="_blank">
						{$i18n.t('To learn more about powerful prompt variables, click here')}
					</a>
				</div>
			</div>
		</div>

		<div class="my-4 flex justify-end pb-20">
			<button
				class=" text-sm w-full lg:w-fit px-4 py-2 transition rounded-xl {loading
					? ' cursor-not-allowed bg-black hover:bg-gray-900 text-white dark:bg-white dark:hover:bg-gray-100 dark:text-black'
					: 'bg-black hover:bg-gray-900 text-white dark:bg-white dark:hover:bg-gray-100 dark:text-black'} flex w-full justify-center"
				type="submit"
				disabled={loading}
			>
				<div class=" self-center font-medium">{$i18n.t('Save & Create')}</div>

				{#if loading}
					<div class="ml-1.5 self-center">
						<Spinner />
					</div>
				{/if}
			</button>
		</div>
	</form>
</div>
