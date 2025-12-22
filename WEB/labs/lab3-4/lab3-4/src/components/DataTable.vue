<template>
  <table>
    <thead>
      <tr>
        <th v-for="col in columns" :key="col">{{ col }}</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="(row, i) in rows" :key="i">
        <td v-for="col in columns" :key="col">{{ row[col] }}</td>
      </tr>
    </tbody>
  </table>
</template>

<script lang="ts">
import { defineComponent, PropType } from "vue";

export default defineComponent({
  name: "DataTable",
  props: {
    columns: {
      type: Array as PropType<string[]>,
      required: true
    },
    rows: {
      type: Array as PropType<Record<string, string | number>[]>,
      required: true
    }
  }
});
</script>

<style scoped>
table {
  border-collapse: collapse;
  width: 100%;
}
th, td {
  border: 1px solid #999;
  padding: 8px;
  text-align: center;
}
th {
  background: #393939;
  color: white;
}
tbody tr:nth-child(odd) { background: #f2f2f2; }
tbody tr:nth-child(even) { background: #dadada; }
</style>